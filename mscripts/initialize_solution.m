function solution = initialize_solution(exps, mode)
    switch mode
        case "base"
            solution = struct('exp',cellstr(exps),'mode', mode,'selection',[], 'A',[], 'b',[], 'acc',[]);
        case "pbi" %prediciton by inference
            shifts = 1000;
            one_shift = struct('shifts',num2cell(1:shifts), 'A',[], 'b',[]);
            solution = struct('exp',cellstr(exps), 'mode', mode, 'selection',[], 'sets', one_shift, 'acc',[]);
    end
end