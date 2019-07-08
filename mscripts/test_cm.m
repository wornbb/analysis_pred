function solution = test_cm(data, solution)
    th = 0.04;
    n = length(data);
    for k = 1:n
        switch solution(k).mode
            case "base"
                pred_val = solution(k).A * data(k).variable.xtest(solution(k).selection,:) + mean(solution(k).b,2);
                y_val = data(k).variable.ytest(data.variable.y_index,:);
            case "pbi"
                forecast_powers = 1;
                data = pbi_test_prepare(data, solution.order, forecast_powers);
                As = cat(3, solution(:).sets.A);
                Axs = batch_mtimes(As, data(k).variable.xtest(solution(k).selection,:,:));
                pAxs = permute(Axs, [3 1 2]);
                predition = pAxs .* solution(k).weight;
                formated_pred = sum(permute(predition, [2 3 1]),3);
                E = data(k).variable.ytest(:,:,end)  -  formated_pred;   

                y_val = data(k).variable.ytest(:,:,end);
                pred_val = formated_pred; 
        end
            
        pred = (pred_val > (1 + th)) | (pred_val < (1 - th));
        y = (y_val > (1 + th)) | (y_val < (1 - th));

        p = sum(pred,'all');
        n = numel(pred) - p;

        buffer = pred - y;
        fp = sum(buffer == 1, 'all');
        tp = p - fp;
        fn = sum(buffer == -1, 'all');
        tn = n - fn;
        heatmap([tp,fp;fn, tn])
    end

end