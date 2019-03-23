root = "/data/yi/vioPred/data";
% define violation
Vth = 0.8;
vioP = 5;
% training
% load data
exps = ["Yaswan2c"];
modes = ["base", "pbi"];
mode = modes(2); 
batch_size = 100;
data = load_exps(exps,1, batch_size, root);
test_ratio = 0.1;
data = split_data(data, test_ratio, "random", 50/100);
% take odd index values as input
% take even index values as prediciton for training
% preprocessing
data = norm_data(data);
solution = initialize_solution(exps, mode);
solution = lasso_select_sensors(data, solution);
shift = 3;
solution = ols_prediction(data, solution, shift);

n = length(data);
x_size = size(data(1).variable.xtest) - [0, shift];
y_size = size(data(1).variable.ytest) - [0, shift];
xtest = zeros([x_size, shift]);
ytest = zeros([y_size, shift]);
for j = 1:n
    for k = 1:shift
        xtest(:,:,k) = data(j).variable.xtest(:, 1 + k: end - shift + k);
        ytest(:,:,k) = data(j).variable.ytest(:, 1 + k: end - shift + k);
        x(:,:,k) = data(j).variable.x(:, 1 + k: end - shift + k);
        y(:,:,k) = data(j).variable.y(:, 1 + k: end - shift + k);
    end
    data(j).variable.xtest = xtest;
    data(j).variable.ytest = ytest;
    data(j).variable.x = x;
    data(j).variable.y = y;
end

for k = 1:n
    As = cat(3, solution(:).sets.A);
    Axs = batch_mtimes(As, data(k).variable.x);
    cvx_begin
        variable weight(shift)
        minimize norm( data(k).variable.y -  Axs .* weight)
    cvx_end
    solution(k).weight = weight;
end

solution = test_sol(data, solution);
%save_solution(solution, root);