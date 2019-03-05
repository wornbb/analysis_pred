function b = batch_lasso_l2(x,y,opt_t)
    t = 20;
    if nargin == 3
        t = opt_t;
    end
    col = size(x,1);
    row = size(y,1);

    cvx_begin quiet
        variable b(row, col)
        minimize( norm(y-b*x, Inf) )
        subject to
            sum(norms(b'),2) <= t
    cvx_end
end
