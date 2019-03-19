function solution = opt_linear(data, solution)
    n = length(data);
    for k = 1:n
        As = cat(3, solution(:).sets.A);
        Axs = batch_mtimes(As, data(k).variable.x(solution(k).selection, :, :));
        estimate = 0;
        cvx_begin
            variable weight(shift)
            for j = 1:shift - 1
                estimate = weight(j) * Axs(:, :, j) + estimate;
            end
            minimize norm( data(k).variable.y - estimate)
        cvx_end
        solution(k).weight = weight;
    end
end