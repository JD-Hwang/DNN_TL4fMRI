function dbn = dbnsetup_rest(dbn, x, opts)

    n = size(x, 2);             % input size
    dbn.sizes = [n, dbn.sizes]; % [input hidden hidden ...]

    for u = 1 : numel(dbn.sizes) - 1
        dbn.rbm{u}.alpha    = opts.alpha(u);           
        dbn.rbm{u}.momentum = opts.momentum;     
        dbn.rbm{u}.activation_function = 'tanh';
        
        % Gaussian-Bernoulii RBM
        if u == 1 && opts.gbrbm == 1
            dbn.rbm{u}.gbrbm = 1;
        else
            dbn.rbm{u}.gbrbm = 0;
        end

        % penalty term
        dbn.rbm{u}.hsparsityParam = 10;   
        dbn.rbm{u}.hsparsityTarget = opts.hsparsityTarget;
        dbn.rbm{u}.wsparsityTarget = opts.wsparsityTarget(u);
        dbn.rbm{u}.weightPenaltyL1 = opts.weightPenaltyL1(u);
        dbn.rbm{u}.weightPenaltyL2 = opts.weightPenaltyL2;
        dbn.rbm{u}.dropoutFraction = opts.dropoutFraction;
        dbn.rbm{u}.hoyerTarget = opts.hoyerTarget(u);

        % progress
        dbn.rbm{u}.error = [];
        dbn.rbm{u}.rho = [];
        dbn.rbm{u}.beta = [];
        dbn.rbm{u}.max_beta = opts.max_beta(u);
        dbn.rbm{u}.mNZR = []; 
        dbn.rbm{u}.lr = [];
        
%         dbn.rbm{u}.beginAnneal = 2000; 
        
%         dbn.rbm{u}.beginAnneal = 1500; % sp 0.9 cmp 10 - 20
%         dbn.rbm{u}.beginAnneal = 600; % sp 0.8 cmp 10 - 20
        dbn.rbm{u}.beginAnneal = 200; % PCA sphering

        dbn.rbm{u}.mHoyer = []; 

        % initialization
        
        fanin = dbn.sizes(u); fanout = dbn.sizes(u + 1);
        dbn.rbm{u}.W  = 2 * sqrt(6/(fanin + fanout)) * (rand(dbn.sizes(u+1), dbn.sizes(u))-0.5); 
%             dbn.rbm{u}.W  = 0.01 * randn(dbn.sizes(u + 1), dbn.sizes(u)); 
        
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));  
        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);    
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);   
        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1); 
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end

end
