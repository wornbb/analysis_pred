n = length(data);
x_size = size(data(1).variable.xtest) - [0, shift];
y_size = size(data(1).variable.ytest) - [0, shift];
xtest = zeros([x_size, shift]);
ytest = zeros([y_size, shift]);
for j = 1:n
    for k = 1:shift
        xtest(:,:,k) = data(j).variable.xtest(:, 1 + k: end - shift + k);
        x(:,:,k) = data(j).variable.x(:, 1 + k: end - shift + k);
        
    end
    data(j).variable.ytest = data(j).variable.ytest(:, 1:end - shift);
    data(j).variable.y = data(j).variable.y(:, 1:end - shift);
    data(j).variable.xtest = xtest;
    data(j).variable.x = x;
end

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
%  
solution = test_sol(data, solution);
