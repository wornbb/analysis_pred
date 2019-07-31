
load("sensor_rank_fig.mat")
[a, rank_1] = test_lasso_rank_sensors(normed_split, solution_select);
function [solution, b_norm] = test_lasso_rank_sensors(data,solution)
    quality_factor = 0.8;
    n = length(data);
    for k = 1:n
        t = 25;
        x = data(k).variable.xbar;
        y = data(k).variable.ybar;
        prob = optimproblem('ObjectiveSense','minimize');
        beta = optimvar('beta', size(y,1), size(x,1), 'Type', 'continuous', 'LowerBound',0); 
        prob.Objective = sum(sum((y - beta * x).*(y - beta * x)));
        lasso = sum(sum(beta)) <= t;
        prob.Constraints.lasso = lasso;
        solution = solve(prob); 
        b_norm = 1;
        %b_norm = norms(b);
        %this might be problem if there are multiple experiment
        
    end
end

