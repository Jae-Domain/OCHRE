"""Compare bisection-control and nonlinear-solver approaches for a 12-node HPWH.

This script uses a two-node predictive model to estimate outlet temperature over
15-minute intervals and then optimizes the next control setpoint based on those
predictions.
Also runs over constant nominal and elevated setpoints

Inputs: Flow data for a given site, ambient temperature, mains temperature, and initial node temperatures.
Outputs: CSV files containing the simulation results for each site and method, as well as a timing metrics CSV file comparing the two methods.
"""

import os
import datetime as dt
import pandas as pd
import numpy as np

from ochre import Dwelling
from ochre.utils import default_input_path  # for using sample files
from ochre import HeatPumpWaterHeater


import os
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint, Bounds
import time

#Global Parameters 
TANK_VOLUME = 151 #40g
MONTH_MIN = 20160*2

# Define equipment and simulation parameters
SETPOINT_DEFAULT = 51.5  # in C #alternate b/w 60 and 49
DEADBAND_DEFAULT = 5.56  # in C

MAX_SETPOINT = 60
MIN_SETPOINT = 49  # minimum setpoint - 40.6, 44.5, and 49

BISECTION_TEMP = 49



def get_rolling_flow_avg(data, current_date, horizon = 2, backlog = 14):# needs to receive previous two weeks of flow data, data should contain previous flow
    # Select current 2-week slice
    day_minutes = 60 * 24
    hour = current_date.hour
    minute = current_date.minute
    is_weekend = current_date.weekday() >= 5 #if is weekend

    two_weeks = data[:day_minutes * backlog] #Determines how many days back to use for data
    two_weeks.columns = ["readTime", "Flow"]
    two_weeks["readTime"] = pd.to_datetime(two_weeks["readTime"])
    two_weeks['hour'] = two_weeks['readTime'].dt.hour
    two_weeks["is_weekend"] = two_weeks["readTime"].dt.weekday >= 5 #check if weekend
    two_weeks = two_weeks.groupby(['hour', 'is_weekend'])['Flow'].mean().reset_index()

    predicted_flow = []
    h_offset = 0
    for i in range(horizon * 60):  # per-minute intervals in number of hours
            total_minutes = minute + i
            h_offset = total_minutes // 60
            current_hour = (hour + h_offset) % 24

            flow = two_weeks.loc[
                (two_weeks['hour'] == current_hour) &
                (two_weeks['is_weekend'] == is_weekend),
                'Flow'
            ]

            if not flow.empty:
                predicted_flow.append(flow.iloc[0] / 60)  # convert hourly flow to per-minute
            else:
                predicted_flow.append(0)  # or np.nan

    return predicted_flow



#n node prediction simulation
def predict_two_node(setpoint, temp_states, temp_amb, temp_mains, draw, duration = 135, tank_volume = TANK_VOLUME, nodes=2, low_power = True, wetbulb = 15, ambient = 20, mains = 7):
    equipment_args = {
        "start_time": dt.datetime(2026, 1, 1, 0, 0), #10292, 90023,  # year, month, day, hour, minute
        "time_res": dt.timedelta(minutes=1),
        "duration": dt.timedelta(minutes = duration),
        "verbosity": 9,  # required to get setpoint and deadband in results
        "save_results": False,  # if True, must specify output_path
        # "output_path": os.getcwd(),        # Equipment parameters
        "Setpoint Temperature (C)": setpoint,
        "Tank Volume (L)": tank_volume,
        "Tank Height (m)": 1.22,
        "UA (W/K)": 2.17,
        "HPWH COP (-)": 4.5,
        "water_nodes": nodes,
        "Low Power HPWH":low_power
    }

    # Create water draw schedule
    times = pd.date_range(
        equipment_args["start_time"],
        equipment_args["start_time"] + equipment_args["duration"],
        freq=equipment_args["time_res"],
        inclusive="left",
    )
    #withdraw_rate = np.random.choice([0, water_draw_magnitude], p=[0.99, 0.01], size=len(times))
    withdraw_rate = draw
    withdraw_rate = withdraw_rate[:len(times)]

    schedule = pd.DataFrame(
        {
            "Water Heating (L/min)": withdraw_rate,
            "Water Heating Setpoint (C)": setpoint,  # Setting so that it can reset
            "Water Heating Deadband (C)": DEADBAND_DEFAULT,  # Setting so that it can reset
            "Zone Temperature (C)": temp_amb,
            "Zone Wet Bulb Temperature (C)": wetbulb,  # Required for HPWH
            "Mains Temperature (C)": temp_mains,
        },
        index=times,
    )

    # Initialize equipment
    hpwh = HeatPumpWaterHeater(schedule=schedule, **equipment_args)



    #Temp initialization
    temp_n1 = temp_states[2] #Node 3
    temp_n2 = temp_states[9] #Node 10
    if nodes == 1:
        hpwh.model.states[:] = np.array([(temp_n1 + temp_n2) / 2])
    elif nodes == 12:
        hpwh.model.states = temp_states
        #hpwh.model.states[:] = np.array([temp_n1, temp_n1, temp_n1, temp_n2, temp_n2, temp_n2, temp_n2, temp_n2, temp_n2, temp_n2, temp_n2, temp_n2])#temp_states
    else:
        hpwh.model.states[:] = np.array([temp_n1, temp_n2])

    # Simulate
    control_signal = {}
    setpoints = []

    try:
        for t in hpwh.sim_times:
            #maintain consistent setpoint for the duration of the simulation
            control_signal = {
                "Setpoint": setpoint
            }

            setpoints.append(setpoint)
            # Run with controls
            _ = hpwh.update(control_signal=control_signal)

        
        df = hpwh.finalize()

        if nodes == 1:
            cols_to_save = [
                        "T_WH1",
                        "Hot Water Outlet Temperature (C)",
             ]
            to_save = df.loc[:, cols_to_save]
            to_save = to_save[:-1]
        
            t_1 = to_save["T_WH1"].values[-1]
            t_2 = t_1  # For single-node simulation, both node temperatures are the same
        
        else:
            cols_to_save = [
                "Hot Water Outlet Temperature (C)",
                "T_WH1",
                "T_WH2"
            ]
            to_save = df.loc[:, cols_to_save]
            to_save = to_save[:-1]

            t_1 = to_save["T_WH1"].values[-1]   
            t_2 = to_save["T_WH2"].values[-1]
            if nodes == 12:                
                t_1 = df["T_WH3"].values[-1]
                t_2 = df["T_WH10"].values[-1]




        #return to_save
        return t_1, t_2, to_save["Hot Water Outlet Temperature (C)"].to_numpy()
    
    except Exception as e:
        # Return penalty value (110 C) for invalid configurations
        # This tells optimizer this setpoint is out of bounds
        print(f"Warning: Simulation failed: for setpoint {setpoint}{str(e)}")
        return temp_n1, temp_n2, np.full(15, 0.0)



import numpy as np
from scipy.optimize import minimize, Bounds



#Penalty Optimization
def penalized_objective(x, T1, T2, Tamb, Tmains, draws, min_target=49.0):
    """
    Evaluates setpoints. If ANY outlet temperature falls below min_target (49 C),
    returns a severe penalty equivalent to overriding setpoints to 60 C.
    """
    # Base objective: Sum of setpoints
    base_cost = np.sum(x)
    
    # Run simulation across all periods
    outlet_temps = constraint_wrapper(x, T1, T2, Tamb, Tmains, draws)
    min_outlet_temp = np.min(outlet_temps)
    
    # Check for violation
    if min_outlet_temp < min_target:
        # Distance of violation (how far below 49 C)
        violation_depth = min_target - min_outlet_temp
        
        # Calculate max cost if all setpoints were 60 C (8 * 60 = 480)
        max_setpoint_cost = len(x) * 60.0
        
        # Base penalty jumps to max possible cost, plus a steep multiplier
        # based on violation depth to push the solver back toward safety
        penalty = (max_setpoint_cost - base_cost) + 1000.0 * (violation_depth ** 2)
        return base_cost + penalty

    return base_cost


# --- 2. Black-Box Simulation Constraint Function ---
def constraint_wrapper(x, T1, T2, Tamb, Tmains, d):
    results = []
    t1 = T1
    t2 = T2
    for i in range(len(x)):
        try:
            t1, t2, result = predict_two_node(x[i], t1, t2, Tamb, Tmains, d[i])
            results.append(result)
        except Exception as e:
            # Return realistic low temp value if model fails
            results.append(np.full(15, -10.0)) 

    return np.hstack(results)


# --- 3. Main Solver Function ---
def solve_nonlinear(current_setpoint, T1, T2, Tamb, Tmains, d):

    start_time = time.perf_counter()            
    
    p = 8  # Time periods
    draws = np.array(d).reshape(8, 15)

    # Initial guess starts at safe high temperature (60 C)
    x0 = np.full(p, max(49, current_setpoint))
    bounds = [(49.0, 60.0) for _ in range(p)]

    # --- 4. Solve Using Powell (Derivative-Free & Handles Penalty Functions Great) ---
    result = minimize(
        lambda x: penalized_objective(x, T1, T2, Tamb, Tmains, draws, min_target=49.0),
        x0,
        method='Powell',  # Powell handles penalty step jumps far better than COBYLA
        bounds=bounds,
        options={'ftol': 1e-2, 'maxfev': 400}
    )

    optimized_setpoints = np.clip(np.round(result.x, 2), 49.0, 60.0)

    # Final Check: If the best solution still yields outlet temp < 49 C, force setpoints to 60 C
    final_temps = constraint_wrapper(optimized_setpoints, T1, T2, Tamb, Tmains, draws)
    
    if np.min(final_temps) < 49.0:
        print(f"Violation detected (Min Outlet Temp: {np.min(final_temps):.2f}°C). Overriding setpoints to 60°C.")
        optimized_setpoints = np.full(p, 60.0)

    print("Optimization Success:", result.success)
    print("Optimized Setpoints:", optimized_setpoints)

    elapsed_nonlinear = time.perf_counter() - start_time
    print(
            "Nonlinear Method completed in {elapsed_nonlinear:.4f} seconds."
    )
    return optimized_setpoints

def bisection_control(temp_states, setpoint_initial, ambient, mains, draw, horizon = 2, sim_nodes=2, n_iter = 5, low_power = False): #performs 5 bisection control iterations
    min_temp = MIN_SETPOINT
    max_temp = MAX_SETPOINT

    #Moving upper and lower bounds for viable setpoints
    upper_bound = max_temp
    lower_bound = min_temp

    setpoint = setpoint_initial
    draw_len = horizon * 60 #Expected draw values over minutes
    if len(draw) < (draw_len):
        draw = np.append(draw, [0] * (draw_len - len(draw))) #if insufficient draw value, fill rest with zeroes
    for iteration in range(n_iter):
        t1, t2, t_out = predict_two_node(setpoint, temp_states, ambient, mains, draw, nodes = sim_nodes, duration = draw_len,  low_power = low_power) #returns outlet temperature
        if (t_out < BISECTION_TEMP).any(): #If any output temperatures fall below 49C, increase setpoint      
            lower_bound = setpoint #setpoint must be above current setpoint
            setpoint = setpoint + (upper_bound - setpoint)/2 #move halfway to upper bound
            if setpoint > max_temp - 0.5:
                setpoint = max_temp
                return setpoint
        else: #setpoint is viable
            if setpoint == MIN_SETPOINT:    
                return setpoint
            upper_bound = setpoint
            setpoint = setpoint - (setpoint - lower_bound)/2
            if setpoint < min_temp + 0.5: #if setpoint is too close to min_temp, return min_temp
                setpoint = min_temp
    return upper_bound


#Methods : bisection, nonlinear, load_shift, setpoint
def simulate_12_node(site_number, method = "bisection", setpoint_default = SETPOINT_DEFAULT, tank_volume = TANK_VOLUME, draw = "rolling", horizon = 2, backlog = 14, mpc_freq = 15, sim_nodes = 2, bisection_iter = 5, low_power = True, wetbulb = 15, ambient = 20, mains = 7):
    df = pd.read_csv(f"ochre\\defaults\\Input Files\\Ecotope_flow\\net_flow_{site_number}.csv", header=None)
    first_line = df.iloc[MONTH_MIN, 0] #get start_time two weeks into dataset
    start_time = dt.datetime.strptime(first_line.split(",")[0], "%Y-%m-%d %H:%M:%S") #parse start_time from first line of csv
    #start_time = dt.datetime.strptime(first_line.split(",")[0], "%m/%d/%Y %H:%M") #parse start_time from first line of csv
    
    simulation_days = len(df) // (24 * 60)  - 30 # Calculate the number of days based on the number of rows in the CSV file    
    #simulation_days = 1
    

    equipment_args = {
        "start_time": start_time,  # year, month, day, hour, minute
        "time_res": dt.timedelta(minutes=1),
        "duration": dt.timedelta(days=simulation_days),
        "verbosity": 9,  # required to get setpoint and deadband in results
        "save_results": False,  # if True, must specify output_path
        # "output_path": os.getcwd(),        # Equipment parameters
        "Setpoint Temperature (C)": setpoint_default,
        "Tank Volume (L)": tank_volume,
        "Tank Height (m)": 1.22,
        "UA (W/K)": 2.17,
        "HPWH COP (-)": 4.5,
        "water_nodes": 12,
        "Low Power HPWH": low_power,
    }

    # Create water draw schedule
    times = pd.date_range(
        equipment_args["start_time"],
        equipment_args["start_time"] + equipment_args["duration"],
        freq=equipment_args["time_res"],
        inclusive="left",
    )

    withdraw_rate = df.iloc[:, 1].to_numpy()  # Assuming the second column contains the flow data
    previous_rate = df.copy() #back up of last two weeks, holds rolling average
    withdraw_rate = withdraw_rate[MONTH_MIN:MONTH_MIN + len(times)] #stagger by two weeks
    perfect_draws = withdraw_rate.copy()


    schedule = pd.DataFrame(
        {
            "Water Heating (L/min)": withdraw_rate,
            "Water Heating Setpoint (C)": setpoint_default,  # Setting so that it can reset
            "Water Heating Deadband (C)": DEADBAND_DEFAULT,  # Setting so that it can reset
            "Zone Temperature (C)": ambient,
            "Zone Wet Bulb Temperature (C)": wetbulb,  # Required for HPWH
            "Mains Temperature (C)": mains,
        },
        index=times,
    )

    # Initialize equipment
    hpwh = HeatPumpWaterHeater(schedule=schedule, **equipment_args)

    #CHANGE THIS LATER< DEBUGGING FOR COLD DRAW   
    #hpwh.model.states[:] = np.array([mains, mains, mains, mains, mains, mains, mains, mains, mains, mains, mains, mains]) #initialize all nodes to mains temperature

    # Simulate
    control_signal = {}
    setpoints = []
    setpoints_predict = [] #setpoints predicted by nonlinear solver
    setpoint = setpoint_default     
    #generate noise for setpoint profile

    # ==================== LOGGING SETUP ====================
    # log_file_path = f"timestep_log_site_{site_number}_{method}_{draw}.csv"
    # log_file = open(log_file_path, "w")
    # # Write header with requested metrics
    # log_file.write("Timestamp,Hot Water Unmet Demand (kWh),Energy (kWh)\n")
    # =======================================================

    draw_len = horizon * 60 #PREDICITON HORIZON 

    for t in hpwh.sim_times:
        # Change setpoint based on hour of day
        #get optimal setpoint
        if method == "load_shifting": #load up from 5 - 6 am, peak hours 5 - 9 pm, load up from 4 to 5
            if (0 <= t.hour <= 6) or t.hour >= 21: #Less use expected
                setpoint = 49
            else:
                setpoint = 60
        
        #Change setpoint every 15 minutes        
        if (t.minute % mpc_freq == 0):

            if method != 'load_shifting' and method != 'constant':
                #Every 15 minutes, get predicted draw for next 2 hours, solve for dynamic setpoint
                if draw == "rolling":
                    predict_draws = get_rolling_flow_avg(previous_rate, t, horizon = horizon, backlog = backlog)
                elif draw == "perfect": #every 15 minutes, we extract the first 15 elements  
                    predict_draws = perfect_draws[0:draw_len] #gets two hour horizon
                    # then remove 15 values
                    perfect_draws = perfect_draws[mpc_freq:]
                if method == "nonlinear":
                    setpoints_predict = solve_nonlinear(setpoint, hpwh.model.next_states[2], hpwh.model.next_states[9], ambient, mains, predict_draws) #get node temperatures from 3 and 10
                    setpoints_predict = setpoints_predict.tolist()
                    print(setpoints_predict)               
                    #update immediate setpoint to first value of predicted setpoints
                    setpoint = setpoints_predict.pop(0)

                elif method == "bisection":                    
                    setpoint = bisection_control(hpwh.model.next_states, setpoint, ambient, mains, predict_draws, sim_nodes=sim_nodes, horizon = horizon, n_iter = bisection_iter, low_power = low_power) #get node temperatures from 3 and 10

                previous_rate = previous_rate[mpc_freq:]#[60 * mpc_freq:] #shuffle previous rate by mpc_freq minutes


            
        control_signal = {
            "Setpoint": setpoint
        }

        setpoints.append(setpoint)

        # Run with controls
        _ = hpwh.update(control_signal=control_signal)

        # Run with controls        


        
    df = hpwh.finalize()


    #Compute kWh
    df["Hot Water Unmet Demand (kWh)"] = df['Hot Water Unmet Demand (kW)'] / 60
    df['Water Heating Electric Energy (kWh)'] = df['Water Heating Electric Power (kW)']/ 60  # energy per minute

    cols_to_save = [
        "Hot Water Outlet Temperature (C)",
        "Hot Water Unmet Demand (kWh)",
        'Water Heating Electric Power (kW)',
        "Water Heating Electric Energy (kWh)",
        "Water Heating Heat Pump COP (-)",
        "Water Heating Mode",
        "T_WH1",
        "T_WH3",
        "T_WH6" ,       
        "T_WH10",
        "T_WH12"
    ]

    to_save = df.loc[:, cols_to_save]
    to_save['Setpoints'] = setpoints
    to_save["Draw Data"] = pd.Series(withdraw_rate, index=to_save.index)

    to_save



    to_save = to_save[1:-1] 
    if method == 'constant':
        if low_power:
            to_save.to_csv(f'output_site_{site_number}_{tank_volume}(L)_{setpoint}_120V.csv', header=True, index=False)    
        else:
            to_save.to_csv(f'output_site_{site_number}_{tank_volume}(L)_{setpoint}_240V.csv', header=True, index=False)    
    else:
        if low_power:
            to_save.to_csv(f'output_site_{site_number}_{method}_{draw}_{sim_nodes}_120V_{horizon}_{backlog}_{mpc_freq}.csv', header=True, index=False)
        else:
            to_save.to_csv(f'output_site_{site_number}_{method}_{draw}_{sim_nodes}_240V.csv', header=True, index=False)



# ================================
# SITES
# ================================ 22096


import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd


#Good test sites 90069 90130, 21578, 90023, 90051, 90131, 90034
#sites = [90069, 90130, 21578, 90023, 90051, 90131, 90034]
sites = [90135, 99084, 99092, 99094, 99103, 99148, 99162]

sites = [13438, 22096, 90069]


sites = [22096, 13438,
     11531, 23744, 11289, 13265, 23666, 
    90028, 90050, 90135, 10441, 90015]

sites = [90030,
    21578, 22897, 90023, 90130, 99094, 90051,
     90131, 90034, 99148, 99162, 99103,
    99092, 99084, 90069, 90023
]

sites_g = {
    60:  [90159, 22096, 13438, 11531, 23744, 11289, 13265, 23666, 90028, 90050, 90135, 10441, 90015, 90030 ],
    80: [21578, 22897, 90023, 90130, 99094, 90051, 90069, 90131, 90034],
    50: [99148, 99162, 99103, 99092, 99084] 
}

sites_g = {
    #40: ['sample']
    60:  [10441],
    80: [90069, 90130, 99094, 21578],
}
for gallon_size in sites_g:
    sites = sites_g[gallon_size]
    for site_number in sites:

        #Run test simulations
        #site_number = 'sample'
    
        #2 node predictor, perfect knowledge
        horizon = 13

        import time

        #Compare 2 and 12 node simulations
        #simulate_12_node(site_number, method = "bisection", draw="perfect", sim_nodes=2, low_power = True, horizon = 13, mpc_freq = 5)
        #simulate_12_node(site_number, method = "bisection", draw="perfect", sim_nodes=2, low_power = True, horizon = 2, mpc_freq = 15)
        #simulate_12_node(site_number, method = "bisection", draw="perfect", sim_nodes=12, low_power = True, horizon = 2, mpc_freq = 15)
        #simulate_12_node(site_number, method = "constant", setpoint_default = 60, low_power = True, horizon = 3)

        #DOE Experiments
        #abC
        simulate_12_node(site_number, method = "bisection", draw="rolling", sim_nodes=2, low_power = True, horizon = horizon, backlog = 7, mpc_freq = 5)
        #Abc
        simulate_12_node(site_number, method = "bisection", draw="rolling", sim_nodes=2, low_power = True, horizon = horizon, backlog = 7, mpc_freq = 15)

        simulate_12_node(site_number, method = "bisection", draw="rolling", sim_nodes=2, low_power = True, horizon = horizon, backlog = 21, mpc_freq = 5)
        #Abc

        simulate_12_node(site_number, method = "bisection", draw="rolling", sim_nodes=2, low_power = True, horizon = horizon, backlog = 21, mpc_freq = 15)
        #ABC
        simulate_12_node(site_number, method = "bisection", draw="rolling", sim_nodes=2, low_power = True, horizon = horizon, backlog = 28, mpc_freq = 5)
        #aBc
        simulate_12_node(site_number, method = "bisection", draw="rolling", sim_nodes=2, low_power = True, horizon = horizon, backlog = 28, mpc_freq = 15)

        simulate_12_node(site_number, method = "bisection", draw="rolling", sim_nodes=12, low_power = True, horizon = 3, backlog = 14, mpc_freq = 15)

        #1-node
        #simulate_12_node(site_number, method = "bisection", draw="perfect", sim_nodes=1, low_power = True, horizon = horizon)
        #simulate_12_node(site_number, method = "bisection", draw="rolling", sim_nodes=1, low_power = True, horizon = horizon)

        #Baselines
        #simulate_12_node(site_number, "constant", setpoint_default = 60, low_power = True) #run at elevated setpoint        
        #simulate_12_node(site_number, "constant", setpoint_default = 49, low_power = True) #run at elevated setpoint
        # simulate_12_node(site_number, "load_shifting", low_power = True)

