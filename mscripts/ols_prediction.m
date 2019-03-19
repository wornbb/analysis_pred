function solution = ols_prediction(data, solution, order, prediction_power)
    n = length(data);
    for k = 1:n
        for j = 1:order
            solution(k).order = order;
            neg_offset = order + prediction_power;
            [solution(k).sets(j).A, ~] = my_ols(data(k).variable.x(solution(k).selection, 1 + j:end - neg_offset + j), data(k).variable.y(:, 1 + j + prediction_power:end - order + j));
        end
    end

end