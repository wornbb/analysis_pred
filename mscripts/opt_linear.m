function solution = opt_linear(data, solution)
    n = length(data);
    for k = 1:n
        As = cat(3, solution(:).sets.A);
        Axs = batch_mtimes(As, data(k).variable.x(solution(k).selection, :, :));
        estimate = 0;
        order = solution(k).order;
        cvx_begin
            variable weight(order)
            for j = 1:order
                estimate = weight(j) * Axs(:, :, j) + estimate;
            end
            minimize norm( data(k).variable.y - estimate)
        cvx_end
        solution(k).weight = weight;
    end
end