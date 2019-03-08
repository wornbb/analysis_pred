% define violation
Vth = 0.8;
vioP = 5;
% training
% load data
exp = ["Yaswan2c"];
batch_size = 1000;
data = load_exps(exp,1, batch_size);
test_ratio = 0.1;
data = split_data(data, test_ratio, "random", 1/100);
% take odd index values as input
% take even index values as prediciton for training
% preprocessing
data = norm_data(data);
solution = lasso_select_sensors(data);

 
n = length(data);
for k = 1:n
    [solution(k).A ,solution(k).b] = my_ols(data(k).variable.x(solution(k).selection,:), data(k).variable.y);
end
disp("half way");
% testing 
%results
n = length(data);
for k = 1:n
    E = data(k).variable.ytest - solution(k).A*data(k).variable.xtest(solution(k).selection,:); %- solution(k).b;
    obj = norm(E);
    disp(["norm of Error: ", obj])

    base = 0.001;
    disp(["total entries: ", size(data(k).variable.ytest,1)*size(data(k).variable.ytest,2)]);
    for j = 1:3:21
        g = base * j;
        good = (abs(E./data(k).variable.ytest) <= g);
        disp(["total correct prediction with acc ", g, ":", sum(sum(good))]);
    end
end



