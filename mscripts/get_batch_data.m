function batch_data = get_batch_data(name, iter, batch_size)
    root = "/data/yi/vioPred/data";
    fname = name + ".gridIR";
    f = fullfile(root, name,  fname);
    fid = fopen(f);
    % load the first line to get the size of matrix
    % so we can initialize batch_data 
    tline = fgetl(fid);
    matrix1D = str2num(tline);
    batch_data = zeros(length(matrix1D), batch_size);
    frewind(fid);
   % skip unwanted cycles
    offset = (iter - 1) * batch_size;
    for k = 1:offset
        fgetl(fid);
    end
    % real loading
    target = iter * batch_size;
    for k = 1:batch_size
        tline = fgetl(fid);
        matrix1D = str2num(tline);
        batch_data(:,k + offset) = matrix1D';
    end
        
    fclose(fid);
end
