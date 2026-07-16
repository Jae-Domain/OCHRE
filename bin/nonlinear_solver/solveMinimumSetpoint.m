% --- 1. Define Problem Parameters & Time-Series Data ---
p = 24; % Example: 24 intervals (6 hours of 15-min intervals)
T_lower_bound = 40.6; % T_min acceptable outlet temp

% Input actual data
T1 = 50 * ones(p,1); 
T2 = 48 * ones(p,1);
Tamb = 20 * ones(p,1);
Tmains = 15 * ones(p,1);
d = randi([0, 5], p, 1); 

% --- 2. Setup Optimization Variables ---
objectiveFcn = @(x) sum(x);
x0 = 55 * ones(p, 1);
lb = 49 * ones(p, 1);
ub = 60 * ones(p, 1);

% --- 3. Package the Nonlinear Constraints ---
nonlinconFcn = @(x) myConstraints(x, T1, T2, Tamb, Tmains, d, T_lower_bound);

% --- 4. Set Options (Optional but recommended) ---
options = optimoptions('fmincon', 'Display', 'iter', 'Algorithm', 'interior-point');

% --- 5. Run fmincon ---
[S_opt, fval, exitflag, output] = fmincon(...
    objectiveFcn, ... % fun
    x0,           ... % x0
    [], [],       ... % A, b
    [], [],       ... % Aeq, beq
    lb, ub,       ... % lb, ub
    nonlinconFcn, ... % nonlcon
    options       ... % options
);

% S_opt now contains your optimized setpoints for each time interval!