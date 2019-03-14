n = length(data);
x_size = size(data(1).variable.xtest) - [0, shift];
y_size = size(data(1).variable.ytest) - [0, shift];
xtest = zeros([x_size, shift]);
ytest = zeros([y_size, shift]);
for j = 1:n
    for k = 1:shift
        xtest(:,:,k) = data(j).variable.xtest(:, 1 + k: end - shift + k);
        ytest(:,:,k) = data(j).variable.ytest(:, 1 + k: end - shift + k);
        x(:,:,k) = data(j).variable.x(:, 1 + k: end - shift + k);
        y(:,:,k) = data(j).variable.y(:, 1 + k: end - shift + k);
    end
    data(j).variable.xtest = xtest;
    data(j).variable.ytest = ytest;
    data(j).variable.x = x;
    data(j).variable.y = y;
end

for k = 1:n
    As = cat(3, solution(:).sets.A);
    Axs = batch_mtimes(As, data(k).variable.x(solution(k).selection, :, :));
    Axs = permute(Axs, [3, 1, 2]);
    cvx_begin
        variable weight(shift,1)
        minimize norm( permute(sum(Axs .* weight,1), [2,3,1]))
    cvx_end
    solution(k).weight = weight;
end
%data(k).variable.y -  
solution = test_sol(data, solution);