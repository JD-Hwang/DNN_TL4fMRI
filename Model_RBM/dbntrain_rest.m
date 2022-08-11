function dbn = dbntrain_rest(dbn, x, opts)

    n = numel(dbn.rbm);
    dbn.rbm{1} = rbmtrain_rest_grp(dbn.rbm{1}, x, opts, 1); 

    %     for i = 2 : n
        %     x = rbmup_rest(dbn.rbm{i - 1}, x, opts);
        %     dbn.rbm{i} = rbmtrain_rest_grp(dbn.rbm{i}, x, opts, i);
    %     end

end
