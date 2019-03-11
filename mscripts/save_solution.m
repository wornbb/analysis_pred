function save_solution(solution, root)
    if nargin ~= 2
        root = "/data/yi/vioPred/data";
    end  
    name = date;
    fname = fullfile(root, name + ".mat");
    save(fname,"solution");
end