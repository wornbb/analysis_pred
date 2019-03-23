function data = pbi_test_prepare(data, order, forecast_power)
% shift x and y so that we can use older x to predict future y.
    n = length(data);
    xt_size = size(data(1).variable.xtest) - [0, order + forecast_power];
    yt_size = size(data(1).variable.ytest) - [0, order + forecast_power];
    xtest = zeros([xt_size, order]);
    x_size = size(data(1).variable.x) - [0, order + forecast_power];
    y_size = size(data(1).variable.y) - [0, order + forecast_power];
    x = zeros([x_size, order]);
    if any([xt_size, x_size]<0)
        error('Too few samples, probably for testing')
    end
    for j = 1:n
        for k = 1:order
            xtest(:,:,k) = data(j).variable.xtest(:, 1 + k: end - forecast_power - order + k);
            x(:,:,k) = data(j).variable.x(:, 1 + k: end - order - forecast_power + k);
        end
        data(j).variable.xtest = xtest;
        data(j).variable.ytest = data(j).variable.ytest(:, 1 + forecast_power: end - order);
        data(j).variable.x = x;
        data(j).variable.y = data(j).variable.y(:, 1 + forecast_power: end - order);
    end
end