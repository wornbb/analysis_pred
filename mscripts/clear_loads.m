function clear_loads(root)
    if ~nargin
        root = "/data/yi/vioPred/data";
    end
    exp_record = fullfile(root, "exp_record.mat");
    exp_save   = fullfile(root, "exp_save.mat");
    delete(exp_record)
    delete(exp_save)
end