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
exps = ["Yaswan2c"];
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
parfor k = 1:length(forecast_powers)
    data = data_shared;
    solution = initialize_solution(exps, mode);
    solution = lasso_select_sensors(data, solution);
    order = 3;
    solution = ols_prediction(data, solution, order, forecast_powers(k));
    data = pbi_test_prepare(data, order, forecast_powers(k));
    solution = opt_linear(data, solution);
    solution = test_sol(data, solution);
    solutions(k).sol = solution;
    %save('temp.mat');
end
save_solution(solutions, root);
