function data = load_exps(exps, iter, batch_size, root)
    if nargin == 3
        root = "/data/yi/voltVio/analysis/raw";
    end    
    % my dirs
    exp_record = fullfile(root, "exp_record.mat");
    exp_save   = fullfile(root, "exp_save.mat");
    % does save files exist?
    try 
        load(exp_record, "old_exp");
        % is it the same as exps?
        try 
            same_exp= all([old_exp(:).name] == exps);
            same_iter = all([old_exp(:).iter] == iter);
            same_batch = all([old_exp(:).batch_size] == batch_size);
            already_saved = same_exp & same_iter & same_batch;
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
        data = struct('index',num2cell(1:n),'variable',struct('exp',[],'x',[],'y',[],'xbar',[],'ybar',[],'xtest',[],'ytest',[],'origin',[],'y_index',[]));
        for k = 1:n
            data(k).variable.exp = exps(k);
            data(k).variable.origin = get_batch_data(exps(k), iter, batch_size, root);
        end
    end
    old_exp = struct('name',cellstr(exps),'iter',iter,'batch_size',batch_size);
    save(exp_record, 'old_exp');
    save(exp_save, 'data');
end