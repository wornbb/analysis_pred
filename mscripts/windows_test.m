root = "C:\Users\wornb\Desktop";
% define violation
Vth = 0.8;
vioP = 5;
% training
% load data
exps = ["Yaswan2c"];
mode = "base";
batch_size = 10;
data = load_exps(exps,1, batch_size, root);
test_ratio = 0.1;
data = split_data(data, test_ratio, "random", 1/100);
% take odd index values as input
% take even index values as prediciton for training
% preprocessing
data = norm_data(data);
solution = initialize_solution(exps, mode);
solution = lasso_select_sensors(data, solution);

solution = ols_inference(data,solution);

solution = test_sol(data, solution, mode);
save_solution(solution, root);