function solution = initialize_solution(exps, mode)
    switch mode
        case "base"
            solution = struct('exp',cellstr(exps),'mode', mode,'selection',[], 'A',[], 'b',[], 'acc',[]);
    end
end