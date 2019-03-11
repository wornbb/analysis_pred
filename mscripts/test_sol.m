function solution = test_sol(data, solution, mode)
    n = length(data);
    base_acc = 0.00001;
    tests = [1:9, 10:10:90, 100:100:1000];
    test_acc = base_acc * tests;
    switch mode
        case "base"
            for k = 1:n
                E = data(k).variable.ytest - solution(k).A*data(k).variable.xtest(solution(k).selection,:); %- solution(k).b;        
                diff = abs(E./data(k).variable.ytest);
                solution(k).acc = batch_cmp(diff, test_acc);
            end
    end
end