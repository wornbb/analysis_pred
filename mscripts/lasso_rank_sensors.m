function [solution, b_norm] = lasso_rank_sensors(data,solution)
    quality_factor = 0.8;
    n = length(data);
    for k = 1:n
        t = 25;
        b = batch_lasso_l2(data(k).variable.xbar, data(k).variable.ybar, t);
        b_norm = norms(b);
        %this might be problem if there are multiple experiment
    end
end