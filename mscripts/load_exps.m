function data = load_exps(exps, iter, batch_size)
    % my dirs
    dump = "/data/yi/vioPred/data";
    exp_record = fullfile(dump, "exp_record.mat");
    exp_save   = fullfile(dump, "exp_save.mat");
    % does save files exist?
    try 
        load(exp_record, "old_exp");
        % is it the same as exps?
        try 
            already_saved = all(old_exp == exps);
        catch % all strange results imply not the same
            already_saved = 0;
        end
    catch 
        already_saved = 0;
    end
    if already_saved
        load(exp_save, "data");
    else
        n = length(exps);
        data = struct('index',num2cell(1:n),'variable',struct('exp',[],'x',[],'y',[],'xbar',[],'ybar',[],'origin',[]));
        for k = 1:n
            data(k).variable.exp = exps(k);
            data(k).variable.origin = get_batch_data(exps(k), iter, batch_size);
        end
    end
end