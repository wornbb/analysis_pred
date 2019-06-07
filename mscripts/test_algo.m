root = "C:\Users\Yi\Desktop";

% training
exps = ["Yaswan2c"];
mode = "base";
batch_size = 4000;
data = load_exps(exps,1, batch_size, root);
test_ratio = 0.01;
data = split_data(data, test_ratio, "random", 1/100);
% take odd index values as input
% take even index values as prediciton for training
% preprocessing
data = norm_data(data);
solution = initialize_solution(exps, mode);
solution = lasso_select_sensors(data, solution);

solution = ols_inference(data,solution);


test_data = get_batch_data('last_5000',0,5000,root);

predict_str = 1;

data.variable.ytest = test_data(:,1 + predict_str:end);
data.variable.xtest = test_data(:,1:end - predict_str);

test_cm(data,solution)