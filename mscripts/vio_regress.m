tic
% define violation
Vth = 0.8;
vioP = 5;

% load data
exp = "Yaswan2c";
batch_size = 1000;
data = get_batch_data(exp,1, batch_size);
% preprocessing
vg_normed = normalize(data);

% take odd index values as input
x = vg_normed(1:2:end,:);
% take even index values as prediciton for training
y = vg_normed(2:2:end,:);
b = batch_lasso_l2(x,y);

toc
