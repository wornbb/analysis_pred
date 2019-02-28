tic
% define violation
Vth = 0.8;
vioP = 5;
% training
% load data
exp = "Yaswan2c";
batch_size = 100;
data = get_batch_data(exp,1, batch_size);
%vg_normed = normalize(data);
% take odd index values as input
X = data(1:2:end,:);
% take even index values as prediciton for training
Y = data(1:2*50:end,:);
% preprocessing
x = normalize(X);
y = normalize(Y);
b = batch_lasso_l2(x,y,15);
b_norm = norms(b);
best_sensor = max(b_norm);
good_sensor = (b_norm >= 0.8 * best_sensor);
[a ,c] = my_ols(X(good_sensor,:), Y);
toc

% testing 
test_data = get_batch_data(exp,10, batch_size);
Xt = test_data(1:2:end,:);
Yt = test_data(1:2*50:end,:);
norm(Yt-Xt*a - c)