function solution = lasso_select_sensors(data,solution)
    quality_factor = 0.8;
    n = length(data);
    for k = 1:n
        b = batch_lasso_l2(data(k).variable.xbar, data(k).variable.ybar, 15);
        b_norm = norms(b);
        best_sensor = max(b_norm);
        solution(k).selection = (b_norm >= quality_factor * best_sensor);
    end
end