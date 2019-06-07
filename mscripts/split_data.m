function data = split_data(data, test_ratio, option, config)
% get the total voltage gird points in the first observation
    [n, o] = size(data(1).variable.origin);
    train_set = (1:o)<(o*(1-test_ratio));
    switch option
        case "random"
        % in this case, config represent a percentage of data which will be
        % used as Y. The rest is X.
        y_index = (rand(n,1)<config);
        for k = 1:length(data) % for all exps
            data(k).variable.y = data(k).variable.origin(y_index, train_set);
            data(k).variable.x = data(k).variable.origin(~y_index, train_set);
            % same as above, will be overwritten
            data(k).variable.ybar = data(k).variable.origin(y_index, train_set);
            data(k).variable.xbar = data(k).variable.origin(~y_index, train_set);
            data(k).variable.ytest = data(k).variable.origin(y_index, ~train_set);
            data(k).variable.xtest = data(k).variable.origin(~y_index, ~train_set);
            data(k).variable.origin = [];
            data(k).variable.y_index = y_index;
        end
        case "flp"
        % in this case, config is the floor plan.
        a = 1;
    end
end