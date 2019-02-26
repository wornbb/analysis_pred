function batch_data = get_batch_data(name, iter, batch_size)
    root = "/data/yi/vioPred/data";
    fname = name + ".gridIR";
    f = fullfile(root, fname);
    fid = fopen(f);
    % load the first line to get the size of matrix
    % so we can initialize batch_data 
    tline = fgetl(fid);
    matrix1D = str2num(tline);
    batch_data = zeros(length(tline), batch_size);
   % skip unwanted cycles
    skip = (iter - 1) * batch_size;
    for k = 1:skip
        fgetl(fid);
    end
    % real loading
    target = iter * batch_size;
    for k = (1 + skip):target
        tline = fgetl(fid);
        matrix1D = str2num(tline);
        batch_data(:,k) = matrix1D;
    end
        
end