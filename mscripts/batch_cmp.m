function result = batch_cmp(left, right)
    n = length(right);
    total = numel(left);
    result = zeros(size(right) + [0 1]);
    for k = 1:n
        result(k, 1) = right(k);
        result(k, 2) = sum(sum(left<right(k))) / total;
    end
end