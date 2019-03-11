function result = batch_cmp(left, right)
    n = length(right);
    total = numel(left);
    result = zeros(size(right));
    for k = 1:n
        result(n) = sum(sum(left<right(n))) / total;
    end
end