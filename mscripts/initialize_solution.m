function solution = initialize_solution(exps, mode)
    switch mode
        case "base"
            solution = struct('exp',cellstr(exps),'mode', mode,'selection',[], 'A',[], 'b',[], 'acc',[]);
        case "pbi" %prediciton by inference
            max_order = 1000;
            orders = struct('order',num2cell(1:max_order), 'A',[], 'b',[]);
            solution = struct('exp',cellstr(exps), 'mode', mode, 'selection',[], 'sets', orders, 'acc',[], 'order', []);
    end
end