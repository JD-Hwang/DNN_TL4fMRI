function x = rbmup(rbm, x, opts)

    sigma = 1;
%     x = x - repmat(mean(x),size(x,1),1); 
%     x = x ./ repmat(std(x), size(x,1),1) * sigma;
    c_cpu = gather(rbm.c');
    w_cpu = gather(rbm.W');
    x = tanh(repmat(c_cpu, size(x, 1), 1) + x * w_cpu / sigma^2); % sample n x node n
    
end



