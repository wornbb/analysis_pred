function data = norm_data(data)
% get the total voltage gird points in the first observation
    n = size(data(1).variable.origin,1);
        for k = 1:length(data) % for all exps
            data(k).variable.ybar = normalize(data(k).variable.ybar);
            data(k).variable.xbar = normalize(data(k).variable.xbar);
        end
end