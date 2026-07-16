# Requisites

import os
import datetime as dt
import pandas as pd
import numpy as np

from ochre import Dwelling
from ochre.utils import default_input_path  # for using sample files
from ochre import HeatPumpWaterHeater


#Global params

TANK_VOLUME = 151 #40g

import os
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

deadband_default = 5.56


def get_rolling_flow_avg(data, current_date, horizon = 2):# needs to receive previous two weeks of flow data, data should contain previous flow
    # Select current 2-week slice
    day_minutes = 60 * 24
    hour = current_date.hour
    minute = current_date.minute
    is_weekend = current_date.weekday() >= 5 #if is weekend

    two_weeks = data[:day_minutes * 14]
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

#2 node simulation
def predict_two_node(setpoint, temp_n1, temp_n2, temp_amb, temp_mains, draw, duration = 15, tank_volume = TANK_VOLUME):
    setpoint_default = setpoint
    equipment_args = {
        "start_time": dt.datetime(2026, 1, 1, 0, 0), #10292, 90023,  # year, month, day, hour, minute
        "time_res": dt.timedelta(minutes=1),
        "duration": dt.timedelta(minutes = duration),
        "verbosity": 9,  # required to get setpoint and deadband in results
        "save_results": False,  # if True, must specify output_path
        # "output_path": os.getcwd(),        # Equipment parameters
        "Setpoint Temperature (C)": setpoint_default,
        "Tank Volume (L)": tank_volume,
        "Tank Height (m)": 1.22,
        "UA (W/K)": 2.17,
        "HPWH COP (-)": 4.5,
        "water_nodes": 2
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
            "Water Heating Setpoint (C)": setpoint_default,  # Setting so that it can reset
            "Water Heating Deadband (C)": deadband_default,  # Setting so that it can reset
            "Zone Temperature (C)": temp_amb,
            "Zone Wet Bulb Temperature (C)": 15,  # Required for HPWH
            "Mains Temperature (C)": temp_mains,
        },
        index=times,
    )

    # Initialize equipment
    hpwh = HeatPumpWaterHeater(schedule=schedule, **equipment_args)

    hpwh.model.states[:] = np.array([temp_n1, temp_n2])

    # Simulate
    data = pd.DataFrame()
    data = {'draw_data' :[], 'setpoint' :[]}
    control_signal = {}
    setpoints = []

    try:
        for t in hpwh.sim_times:
            # Change setpoint based on hour of day
            setpoint = setpoint_default
            control_signal = {
                "Setpoint": setpoint
            }

            setpoints.append(setpoint)
            # Run with controls
            _ = hpwh.update(control_signal=control_signal)

        
        df = hpwh.finalize()

        cols_to_save = [
            "Hot Water Outlet Temperature (C)",
            "T_WH1",
            "T_WH2"
        ]

        to_save = df.loc[:, cols_to_save]
        to_save = to_save[:-1]
        t_1 = to_save["T_WH1"].values[-1]   
        t_2 = to_save["T_WH2"].values[-1]

        #return to_save
        return t_1, t_2, to_save["Hot Water Outlet Temperature (C)"].to_numpy()
    
    except Exception as e:
        # Return penalty value (110 C) for invalid configurations
        # This tells optimizer this setpoint is out of bounds
        print(f"Warning: Simulation failed for setpoint {setpoint_default}C: {str(e)}")
        return pd.Series([110.0] * 15)



# --- 3. Objective Function ---
def objective(x):
    return sum(x)  # Minimize the sum of setpoints

# --- 4. Black-Box Nonlinear Constraint Function ---
def constraint_wrapper(x, T1, T2, Tamb, Tmains, d):
    results = []
    t1 = T1
    t2 = T2
    for i in range(len(x)): #for each 15 minute time period
        try:
            t1, t2, result = predict_two_node(x[i], t1, t2, Tamb, Tmains, d[i]) #run a simulation for each 15 minute time period, passing in the current setpoint and the previous node temperatures
            results.append(result)
        except Exception as e:
            print(f"Simulation {i} failed: {str(e)}")
            results.append(pd.Series([110.0] * 15))

    return np.hstack(results)

#into nonlinear solver, pass current setpoint (for each 15 minute interval) solve_nonlinear(current_setpoint=50.0, T1=50, T2=50, Tamb=[[20.0]*15]*8, Tmains=[[20.0]*15]*8, d=[[2.5]*15]*8)
#   T1/ T2 - currrent temp, Tamb, Tmains, d - 8x15 arrays of ambient temp, mains temp, and draw for each 15 minute interval
def solve_nonlinear(current_setpoint, T1, T2, Tamb, Tmains, d):
    # Define the constraint: T_out must be between 40.6 and 100
    # --- 2. Parameters ---
    T_lower_bound = 40.6
    p = 8 #time periods
    x0 = [current_setpoint] * p  # Initial guess for setpoints

    #reshape draw values to 8x15 array
    draws = np.array(d).reshape(8, 15)

    # Pass extra arguments via a lambda function. 
# Also make sure to pass 'p' so it's accessible inside the wrapper!
    nl_constraint = NonlinearConstraint(
        lambda x: constraint_wrapper(x, T1, T2, Tamb, Tmains, draws), 
        lb=T_lower_bound, 
        ub=100
    )

    # --- 5. Setpoint Bounds & Setup ---
    bounds = [(49.0, 60.0) for _ in range(p)]

    # --- 6. Solve Using a True Black-Box Method ---
    result = minimize(
        objective,
        x0,
        method='SLSQP',  # Derivative-free trust-region SQP
        bounds=bounds,
        constraints=nl_constraint
    )

    print("Success:", result.success)
    print("Optimized Setpoints:", np.round(result.x, 2))
    return np.round(result.x, 2)  # Return the optimized setpoints

#Global Parameters 
sites = ['90030']
two_weeks = 20160
# Define equipment and simulation parameters
setpoint_default = 51.5  # in C #alternate b/w 60 and 49
deadband_default = 5.56  # in C

max_setpoint = 60
min_setpoint = 49 #minimum setpoint - 40.6, 44.5, and 49
tank_volume = 151 #40g
water_nodes = 12

for site_number in sites:
    
    df = pd.read_csv("ochre\\defaults\\Input Files\\Ecotope_flow\\net_flow_90030.csv", header=None)
    first_line = df.iloc[two_weeks, 0] #get start_time two weeks into dataset
    start_time = dt.datetime.strptime(first_line.split(",")[0], "%Y-%m-%d %H:%M:%S")

    simulation_days = len(df) // (24 * 60)  - 21 # Calculate the number of days based on the number of rows in the CSV file
    time_interval = 2 # adjust setpoint every 2 hours

    equipment_args = {
        "start_time": start_time,  # year, month, day, hour, minute
        "time_res": dt.timedelta(minutes=1),
        "duration": dt.timedelta(days=simulation_days),
        "verbosity": 9,  # required to get setpoint and deadband in results
        "save_results": False,  # if True, must specify output_path
        # "output_path": os.getcwd(),        # Equipment parameters
        "Setpoint Temperature (C)": setpoint_default,
        "Tank Volume (L)": 250,
        "Tank Height (m)": 1.22,
        "UA (W/K)": 2.17,
        "HPWH COP (-)": 4.5,
        "water_nodes": water_nodes
    }

    # Create water draw schedule
    times = pd.date_range(
        equipment_args["start_time"],
        equipment_args["start_time"] + equipment_args["duration"],
        freq=equipment_args["time_res"],
        inclusive="left",
    )

    withdraw_rate = df.iloc[:, 1].to_numpy()  # Assuming the second column contains the flow data
    previous_rate = df.copy() #back up of last two weeks
    withdraw_rate = withdraw_rate[two_weeks:two_weeks + len(times)] #stagger by two weeks
    ambient = 20
    mains = 7

    schedule = pd.DataFrame(
        {
            "Water Heating (L/min)": withdraw_rate,
            "Water Heating Setpoint (C)": setpoint_default,  # Setting so that it can reset
            "Water Heating Deadband (C)": deadband_default,  # Setting so that it can reset
            "Zone Temperature (C)": ambient,
            "Zone Wet Bulb Temperature (C)": 15,  # Required for HPWH
            "Mains Temperature (C)": mains,
        },
        index=times,
    )

    # Initialize equipment
    hpwh = HeatPumpWaterHeater(schedule=schedule, **equipment_args)

    # Simulate
    data = pd.DataFrame()
    data = {'draw_data' :[], 'setpoint' :[]}
    control_signal = {}
    setpoints = []
    setpoints_predict = [] #setpoints predicted by nonlinear solver
    setpoint = setpoint_default     
    #generate noise for setpoint profile
    for t in hpwh.sim_times:
        # Change setpoint based on hour of day
        #get optimal setpoint

        #Change setpoint every 15 minutes        
        if (t.minute % 15 == 0):

            #Every 15 minutes, get predicted draw for next 2 hours, solve for dynamic setpoint
            predict_draws = get_rolling_flow_avg(previous_rate, t)
            setpoints_predict = solve_nonlinear(setpoint, hpwh.model.next_states[2], hpwh.model.next_states[9], ambient, mains, predict_draws) #get node temperatures from 3 and 10
            setpoints_predict = setpoints_predict.tolist()
            print(setpoints_predict)
            previous_rate = previous_rate[60 * 15:] #shuffle previous rate by 15 minutes

            #update immediate setpoint to first value of predicted setpoints
            setpoint = setpoints_predict.pop(0)
            control_signal = {
                "Setpoint": setpoint
            }

            setpoints.append(setpoint)

        # Run with controls
        _ = hpwh.update(control_signal=control_signal)

        
    df = hpwh.finalize()

    cols_to_plot = [
        "Hot Water Outlet Temperature (C)",
        "Hot Water Average Temperature (C)",
        "Water Heating Deadband Upper Limit (C)",
        "Water Heating Deadband Lower Limit (C)",
        "Water Heating Electric Power (kW)",
        "Hot Water Unmet Demand (kW)",
        "Hot Water Delivered (L/min)",
    ]

    cols_to_save = [
        "Hot Water Outlet Temperature (C)",
        #"T_WH1",
        #"T_WH2"
        "T_WH3",
        #"T_WH7",
        "T_WH10",
        #"T_WH12",
        #"T_AMB",
        "Water Heating Heat Pump COP (-)"
    ]

    #avg_setpoints = np.convolve(setpoints, np.ones(15)/15, 'same')
    #avg_setpoints = avg_setpoints[14::15]

    df['Energy (kWh)'] = df['Water Heating Electric Power (kW)']/ 60  # energy per minute
    kwh_energy = df['Energy (kWh)'].resample('15T').sum()  # sum up 15 mins = total kWh per interval

    
    # Convert to kWh per minute
    df['HotWaterDelivered_kWh'] = df['Hot Water Delivered (W)'] / 1000 / 60
    df['HotWaterLoss_kWh'] = df['Hot Water Heat Loss (W)'] / 1000 / 60
    df["HotWaterInjection_kWh"] = df['Hot Water Heat Injected (W)'] / 1000 / 60
    df["Hot Water Unmet Demand (kWh)"] = df['Hot Water Unmet Demand (kW)'] / 60

    # Resample to 15-min totals
    delivered_15min = df['HotWaterDelivered_kWh'].resample('15T').sum()
    loss_15min = df['HotWaterLoss_kWh'].resample('15T').sum()
    injection_15min = df['HotWaterInjection_kWh'].resample('15T').sum()
    unmet_demand_15min = df['Hot Water Unmet Demand (kWh)'].resample('15T').sum()



    # For the DataFrame, select columns and calculate the rolling average for each column
    to_save = df[cols_to_save].rolling(window=15).mean()

    #electric_energy_kwh = electric_energy_kwh[14::15]
    #resample hot water delivered 


    to_save = df.loc[:, cols_to_save]

    to_save["Water Heating Mode"] = df["Water Heating Mode"]

    to_save = to_save[14::15].copy()

    to_save["Water Heating Electric Power"] = kwh_energy.values
    to_save["Hot Water Delivered (kWh)"] = delivered_15min.values
    to_save["Hot Water Heat Loss (kWh)"] = loss_15min.values
    to_save["Hot Water Heat Injected (kWh)"] = injection_15min.values
    to_save["Hot Water Unmet Demand (kWh)"] = unmet_demand_15min.values



    #to_save["Water Heating Electric Power"] = kwh_energy #pd.Series(kwh_energy, index=to_save.index)
    to_save["Setpoints"] = setpoints 

    to_save = to_save[:-1] 
    to_save.to_csv(f'output_site_{site_number}_nonlinear_rolling avg.csv', header=True, index=False)