root = "/data/yi/vioPred/data";
clear_loads(root);
% training 
step_per_cycle = 5;
step_skips = 1:4;
cycle_skips = step_skips * step_per_cycle;
forecast_powers = [step_skips, cycle_skips];
solutions = struct('index',num2cell(1:length(forecast_powers)),'sol',[]);
% define violation
Vth = 0.8;
vioP = 5;
% training
% load data
exps = ["blackscholes2c"];
modes = ["base", "pbi"];
mode = modes(2); 
batch_size = 50;
data = load_exps(exps,1, batch_size, root);
test_ratio = 0.5;
interest_y_ratio = 0.5;
data = split_data(data, test_ratio, "random", interest_y_ratio);
% take odd index values as input
% take even index values as prediciton for training
% preprocessing
data_shared = norm_data(data);
t = [1,10,50,100,1000,10000];
forecast_powers = 5;
parfor k = 1:length(t)
    data = data_shared;
    t_k = t(k)
    solution = initialize_solution(exps, mode);
    solution = lasso_select_sensors(data, solution, t);
    order = 3;
    solution = ols_prediction(data, solution, order, forecast_powers);
    data = pbi_test_prepare(data, order, forecast_powers);
    solution = opt_linear(data, solution);
    solution = test_sol(data, solution);
    solutions(k).sol = solution;
    %save('temp.mat');
end
save_solution(solutions, root);
