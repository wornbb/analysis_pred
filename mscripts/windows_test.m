% define violation
Vth = 0.8;
vioP = 5;

% load data
dim = 100; % dimension needs to be even number
volt_grid = magic(dim);

% take odd index values as input
X = volt_grid(1:2:end,:);
% take even index values as prediciton for training
Y = volt_grid(2:2:end,:);

%preprocessing
x = normalize(X);
y = normalize(Y);
% lasso for sensor selection
b = batch_lasso_l2(x,y);
a = mvregress(X,Y,'algorithm','cwls');

