function solution = ols_prediction(data, solution, shift)
    n = length(data);
    for k = 1:n
        for j = 1:shift
            [solution(k).sets(j).A, ~] = my_ols(data(k).variable.x(solution(k).selection,1:end-shift), data(k).variable.y(:,1+j:end-shift+j));
        end
    end

end