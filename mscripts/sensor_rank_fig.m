
% root = "C:\Users\Yi\Desktop";
% clear_loads(root);
% % training
% step_per_cycle = 5;
% step_skips = 1:4;
% cycle_skips = step_skips * step_per_cycle;
% forecast_powers = [step_skips, cycle_skips];
% solutions = zeros(size(forecast_powers));
% 
% % define violation
% Vth = 0.8;
% vioP = 5;
% % training
% % load data
% exps = ["Yaswan2c"];
% modes = ["base", "pbi"];
% mode = modes(2); 
% batch_size = 50;
% data = load_exps(exps,1, batch_size, root);
% test_ratio = 0.5;
% interest_y_ratio = 0.5;
% split_1 = split_data(data, test_ratio, "random", interest_y_ratio);
% % take odd index values as input
% % take even index values as prediciton for training
% % preprocessing
% normed_split = norm_data(split_1);
% solution_select = initialize_solution(exps, mode);
% [solution_select, rank_1] = lasso_rank_sensors(normed_split, solution_select);
% best_sensor = max(rank_1);
% 
% forecast_powers = 5;
% solutions = zeros(size(3:9));
% solutions = struct('index',num2cell(1:length(1:7)),'sol',[]);
% save('sensor_rank_fig.mat')
for index = 1:3:7
    quality_factor = 0.1 * (index + 2);
    solutions(index).sol = solution_select;
    solutions(index).sol.selection = (rank_1 >= quality_factor * best_sensor);
    order = 3;
    solutions(index).sol = ols_prediction(normed_split, solutions(index).sol, order, forecast_powers);
    data = pbi_test_prepare(normed_split, order, forecast_powers);
    solutions(index).sol = opt_linear(data, solutions(index).sol);
    solutions(index).sol = test_sol(data, solutions(index));
end
