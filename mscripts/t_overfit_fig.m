
root = "C:\Users\Yi\Desktop";
clear_loads(root);
% training
step_per_cycle = 5;
step_skips = 1:4;
cycle_skips = step_skips * step_per_cycle;
forecast_powers = [step_skips, cycle_skips];
solutions = zeros(size(forecast_powers));

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
split_1 = split_data(data, test_ratio, "random", interest_y_ratio);
% take odd index values as input
% take even index values as prediciton for training
% preprocessing
normed_split = norm_data(split_1);
solution_1 = initialize_solution(exps, mode);
[solution_1, rank_1] = lasso_rank_sensors(normed_split, solution_1);

interest_y_ratio = 0.2;
split_2 = split_data(data, test_ratio, "random", interest_y_ratio);
% take odd index values as input
% take even index values as prediciton for training
% preprocessing
normed_split = norm_data(split_2);
solution_2 = initialize_solution(exps, mode);
[solution_2, rank_2] = lasso_rank_sensors(normed_split, solution_2);

boxplot([rank_1 rank_2], [zeros(size(rank_1)), ones(size(rank_2))],'Labels',{'50%','20%'})
ylabel('Importance factor')
xlabel('Percentage of area available compared to total die area for sensor placement')
%save_solution(solution, root);