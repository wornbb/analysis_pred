function product = batch_mtimes(a, b)
    size_a = size(a);
    size_b = size(b);
    if size_a(3) ~= size_b(3)
        error("imcompatible sizes of a and b")
    end
    product = zeros(size_a(1), size_b(2), size_a(3));
    for k = 1:size_a(3)
        product(:,:,k) = a(:,:,k) * b(:,:,k);
    end
end