function solution = ols_inference(data, solution)
% the solution can only inference temporary voltage of interests. Not
% prediciton
    n = length(data);
    for k = 1:n
        [solution(k).A ,solution(k).b] = my_ols(data(k).variable.x(solution(k).selection,:), data(k).variable.y);
    end
end