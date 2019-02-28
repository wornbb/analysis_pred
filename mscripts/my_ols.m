function [b, C] = my_ols(x,y)
    row = size(y, 1);
    col = size(x, 1);
    cvx_begin
        variable b(row, col)
        minimize( norm(y - b*x) )
    cvx_end
    C = y - b*x;
end