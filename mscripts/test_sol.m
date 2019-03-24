function solution = test_sol(data, solution)
    n = length(data);
    base_acc = 0.00001;
    tests = [1:9, 10:10:90, 100:100:1000];
    test_acc = base_acc * tests;
    for k = 1:n
        switch solution(k).mode
            case "base"
                E = data(k).variable.ytest - solution(k).A * data(k).variable.xtest(solution(k).selection,:); %- solution(k).b;        
                diff = abs(E./data(k).variable.ytest);
                solution(k).acc = batch_cmp(diff, test_acc);
            case "pbi"
                As = cat(3, solution(:).sets.A);
                Axs = batch_mtimes(As, data(k).variable.xtest(solution(k).selection,:,:));
                pAxs = permute(Axs, [3 1 2]);
                predition = pAxs .* solution(k).weight;
                formated_pred = permute(predition, [2 3 1]);
                E = data(k).variable.ytest  -  formated_pred; %- solution(k).b;        
                diff = abs(E./data(k).variable.ytest);
                solution(k).acc = batch_cmp(diff, test_acc);
        end
    end

end