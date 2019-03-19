
root = "C:\Users\wornb\Desktop";
clear_loads(root);
% define violation
Vth = 0.8;
vioP = 5;
% training
% load data
exps = ["Yaswan2c"];
modes = ["base", "pbi"];
mode = modes(2); 
batch_size = 40;
data = load_exps(exps,1, batch_size, root);
test_ratio = 0.1;
data = split_data(data, test_ratio, "random", 50/100);
% take odd index values as input
% take even index values as prediciton for training
% preprocessing
data = norm_data(data);
solution = initialize_solution(exps, mode);
solution = lasso_select_sensors(data, solution);

order = 3;
forecast_power = 10;

solution = ols_prediction(data, solution, order, forecast_power);
data = pbi_test_prepare(data, order, forecast_power);
solution = opt_linear(data, solution);
solution = test_sol(data, solution);
%save_solution(solution, root);