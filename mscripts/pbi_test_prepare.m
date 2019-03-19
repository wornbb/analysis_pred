function data = pbi_test_prepare(data, order, forecast_power)
% shift x and y so that we can use older x to predict future y.
    n = length(data);
    xt_size = size(data(1).variable.xtest) - [0, order + forecast_power];
    yt_size = size(data(1).variable.ytest) - [0, order + forecast_power];
    xtest = zeros([xt_size, order]);
    ytest = zeros([yt_size, order]);
    x_size = size(data(1).variable.x) - [0, order + forecast_power];
    y_size = size(data(1).variable.y) - [0, order + forecast_power];
    x = zeros([x_size, order]);
    y = zeros([y_size, order]);
    for j = 1:n
        for k = 1:order
            xtest(:,:,k) = data(j).variable.xtest(:, 1 + k: end - forecast_power - order + k);
            ytest(:,:,k) = data(j).variable.ytest(:, 1 + k + forecast_power: end - order + k);
            x(:,:,k) = data(j).variable.x(:, 1 + k: end - order - forecast_power + k);
            y(:,:,k) = data(j).variable.y(:, 1 + k + forecast_power: end - order + k);
        end
        data(j).variable.xtest = xtest;
        data(j).variable.ytest = ytest;
        data(j).variable.x = x;
        data(j).variable.y = y;
    end
end