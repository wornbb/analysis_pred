% define violation
Vth = 0.8;
vioP = 5;

% load data
dim = 100; % dimension needs to be even number
volt_grid = magic(dim);
% preprocessing
vg_normed = normalize(volt_grid);

% take odd index values as input
x = vg_normed(1:2:end,:);
% take even index values as prediciton for training
y = vg_normed(2:2:end,:);
b = batch_lasso_l2(x,y);

