function rbm = rbmtrain_rest_grp(rbm, x, opts, layer)
    
    h = size(rbm.c, 1);                 % hidden size
    n = size(x, 2);                     % input size
    m = size(x, 1);                     % # of samples
    numbatches = m / opts.batchsize;    % # of batches
            
    epsilon = 0.001;                    % weight sparsity    
    betarate = 0.01;         
    %% all
%     p = 0;
    %% one-by-one
    p = zeros(h,1);
    
%     if rbm.gbrbm == 1                   % gaussian rbm    
%         sigma = 1;
%         x = x - repmat(mean(x),size(x,1),1);
%         x = x ./ repmat(std(x),size(x,1),1) * sigma;
%     else
%         sigma = 1;
%     end
    sigma = 1;

    assert(isfloat(x), 'x must be a float');   
%     assert(rem(numbatches, 1) == 0, 'numbatches not integer');  
    x(isnan(x)) = 0;
        
    for i = 1 : opts.numepochs
        kk = randperm(m);	% random sample index
        err = 0;            % errors in a epoch
        batch_rho = [];     % sparsity
        
        batch_mnzr =[];
        batch_hoyer =[];
        batch_err =[];   
        
        for l = 1 : numbatches
            batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);    % x to batch    
            batch = gpuArray(batch);% batch as gpu array
            
            % dropout
            if rbm.dropoutFraction > 0  
                dropout_mask = (rand(opts.batchsize, h) > rbm.dropoutFraction);
            end

            switch rbm.activation_function
                case 'sigm'
                    v1 = batch;     
                    h1p = sigm(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W' / sigma^2); 
                        if rbm.hsparsityTarget > 0,  rho = mean(h1p, 1); end
                        if rbm.dropoutFraction > 0, h1p = h1p .* dropout_mask; end
                    h1 = double(h1p) > rand(size(h1p));
                    if rbm.gbrbm == 1
                        v2 = repmat(rbm.b', opts.batchsize, 1) + double(h1) * rbm.W; % mean-field
%                         v2 = repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W + sigma^2 * randn(opts.batchsize, n);
                    else
                        v2 = sigm(repmat(rbm.b', opts.batchsize, 1) + double(h1) * rbm.W); 
                    end
                    h2 = sigm(repmat(rbm.c', opts.batchsize, 1) + v2 * rbm.W' / sigma^2);    
                        if rbm.dropoutFraction > 0, h2 = h2 .* dropout_mask; end 
                case 'tanh'
                    v1 = batch;
                     % denoising
                     if opts.denoiselv > 0
                         if layer == 1
                             for bi = 1:opts.batchsize
                                 v1(bi,randsample(n,round(n*opts.denoiselv,0))) = 0;
                             end
                         end
                     end
%
                    h1p = tanh(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W' / sigma^2);  
                        if rbm.hsparsityTarget > 0,  rho = mean(h1p, 1); end
                        if rbm.dropoutFraction > 0, h1p = h1p .* dropout_mask; end
                        h1 = h1p;
                    if rbm.gbrbm == 1
                          v2 = repmat(rbm.b', opts.batchsize, 1) + double(h1) * rbm.W;
%                         v2 = repmat(rbm.b', opts.batchsize, 1) + double(h1) * rbm.W + sigma^2 * randn(opts.batchsize, n);
                    else
                        v2 = tanh(repmat(rbm.b', opts.batchsize, 1) + double(h1) * rbm.W);  
                    end
                    h2 = tanh(repmat(rbm.c', opts.batchsize, 1) + v2 * rbm.W' / sigma^2); 
                        if rbm.dropoutFraction > 0, h2 = h2 .* dropout_mask; end     
                case 'relu'
                    v1 = batch;  
                        s1 = repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W' / sigma^2; 
                    h1p = max(0, s1 + sqrt(sigm(s1)) .* randn(size(s1)));
                        if rbm.hsparsityTarget > 0,  rho = mean(h1p, 1); end
                        if rbm.dropoutFraction > 0, h1p = h1p .* dropout_mask; end
                        h1 = h1p;
                    v2 = repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W;
%                     v2 = repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W + sigma^2 * randn(opts.batchsize, n);
                        s2 = repmat(rbm.c', opts.batchsize, 1) + v2 * rbm.W' / sigma^2;       
                    h2 = max(0, s2 + sqrt(sigm(s2)) .* randn(size(s2)));    
                        if rbm.dropoutFraction > 0, h2 = h2 .* dropout_mask; end    
            end
            
            % update term       
            c1 = h1p' * batch;  % h1p*v1 for the positive phase update. h1 or h1p?
            c2 = h2' * v2;  % h2*v2 for the negative phase update
            vW = rbm.alpha * (c1 - c2)      /sigma^2    / opts.batchsize;    % vW
            vb = rbm.alpha * sum(batch - v2)'	/sigma^2    / opts.batchsize;    % vb
            vc = rbm.alpha * sum(h1p - h2)'             / opts.batchsize;    % vc
            
            % hidden sparsity
            if rbm.hsparsityTarget > 0	
                batch_rho = [batch_rho mean(rho)];
            end
            
            % weight penalty L1
            if rbm.weightPenaltyL1 > 0
                dW = rbm.alpha * rbm.weightPenaltyL1 * sign(rbm.W);
                vW = vW - dW;
            end       
            
            % weight penalty L2
            if rbm.weightPenaltyL2 > 0
                dW = rbm.alpha * rbm.weightPenaltyL2 * rbm.W;
                vW = vW - dW;
            end   
            
            % calcualte sparsity
            for k=1:size(rbm.W,1)
                mHoyer(k) = gather(hoyer(rbm.W(k,:)));
                mNZR(k) = length( find( abs(rbm.W(k,:)) > epsilon ) ) / numel(rbm.W(k,:));
            end
            
            % weight sparsity
            if rbm.wsparsityTarget > 0 || rbm.hoyerTarget > 0
               
                %% one-by-one hoyer
                for k=1:size(rbm.W,1)
                    mNZR(k) = length( find( abs(rbm.W(k,:)) > epsilon ) ) / numel(rbm.W(k,:));
                    mHoyer(k) = gather(hoyer(rbm.W(k,:)));
                    p(k) = p(k) + betarate*sign(rbm.hoyerTarget - mHoyer(k));
                    if p(k) > rbm.max_beta
                            p(k) = rbm.max_beta;
                    elseif p(k) < 0
                        p(k) = 0;
                    end
%                     dW(k,:) = rbm.alpha * p(k) * sign(rbm.W(k,:));
                    dW(k,:) = gather(rbm.alpha * p(k) * sign(rbm.W(k,:)));
                end
                vW = vW - dW;


            end
                        
            rbm.vW = rbm.momentum * rbm.vW + vW;   
            rbm.vb = rbm.momentum * rbm.vb + vb;
            rbm.vc = rbm.momentum * rbm.vc + vc;
            rbm.W = rbm.W + rbm.vW;
            rbm.b = rbm.b + rbm.vb;
            rbm.c = rbm.c + rbm.vc; 
                        
            err = err + sum(sum((batch - v2) .^ 2)) / opts.batchsize;  % error. plz consider batch size.
            
            if mod(l,10) == 0 % l to 10 
                %tmp_err = err / l;
                
                batch_mnzr = [batch_mnzr mNZR];
                batch_hoyer = [batch_hoyer; mHoyer];
                batch_err = [batch_err err/l];
                
                f1=figure();
                set(gcf,'Visible', 'off');
                plot(batch_mnzr);
                ylabel('nzr');
                xlabel('batch');
                ylim([0,1]);
                grid on;
                title(sprintf("%.3f",mean(mNZR)))
                saveas(f1,strcat(opts.savedir,'/layer',num2str(layer),'/epoch',num2str(i),'_nzr.png'));

                f2=figure();
                set(gcf,'Visible', 'off');
                plot(batch_hoyer);
                ylabel('hsp');
                xlabel('batch');
                ylim([0,1]);
                grid on;
                title(sprintf("%.3f",mean(mHoyer)))
                saveas(f2,strcat(opts.savedir,'/layer',num2str(layer),'/epoch',num2str(i),'_hsp.png'));


                f3=figure();
                set(gcf,'Visible', 'off');
                plot(batch_err);
                legend({'Reconstruction error'});
                xlabel('batch');
                grid on;
                saveas(f3,strcat(opts.savedir,'/layer',num2str(layer),'/epoch',num2str(i),'_error.png'));

                close all
            end  
            
            disp([num2str(l) 'th batch ' 'hoyer ' num2str(mean(mHoyer)) datestr(now, ' HH:MM:SS')]);
        end
        
        % hidden sparsity
        if rbm.hsparsityTarget > 0	
            epoch_rho = mean(batch_rho);
            rbm.rho = [rbm.rho epoch_rho];
            rbm.c = rbm.c - rbm.hsparsityParam * (epoch_rho-rbm.hsparsityTarget)';
            disp(['sparsity ' num2str(epoch_rho)]);
        end
        
        % weight sparsity progress
        if rbm.wsparsityTarget > 0 || rbm.hoyerTarget > 0
            %% all
            rbm.mNZR = [rbm.mNZR mNZR]; % nzr
%             rbm.mHoyer = [rbm.mHoyer mHoyer]; % hoyer
%             rbm.beta = [rbm.beta p];
            %% one-by-one
%             rbm.mNZR = [rbm.mNZR; mNZR];
            rbm.mHoyer = [rbm.mHoyer; mHoyer];
            rbm.beta = [rbm.beta; p];      
            
%             disp(['non-zero ratio ' num2str(mNZR)]);
%             disp(['non-zero ratio ' num2str(mean(mNZR))]);
%             disp(['hoyer ' num2str(mHoyer)]);
            disp(['hoyer ' num2str(mean(mHoyer))]);
        end
        
        % Annealing
        if i > rbm.beginAnneal && rbm.beginAnneal ~= 0
            decayrate = -0.000005;    
            rbm.alpha = max( 1e-5, ( decayrate*i+(1-decayrate*rbm.beginAnneal) ) * rbm.alpha ); 
            rbm.lr = [rbm.lr rbm.alpha];
        end
        rbm.lr = [rbm.lr rbm.alpha];
        
%         disp([num2str(rbm.alpha) ' ' num2str(p) ' epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Reconstruction error is: ' num2str(err / numbatches)]); 
        disp([num2str(rbm.alpha) ' ' num2str(mean(p)) ' epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Reconstruction error is: ' num2str(err / numbatches)]); 
        rbm.error = [rbm.error (err/numbatches)];
        
        % Save every 1 epoch
        if mod(i,1)==0 
            
            if ~exist(strcat(opts.savedir,'/rbm'))
                mkdir(strcat(opts.savedir,'/rbm'));
            end
            rbm.W = gather(rbm.W);
            rbm.c = gather(rbm.c);
            save(strcat(opts.savedir,'/rbm/',num2str(layer),'_',num2str(i),'epoc.mat'),'rbm');
            %save(strcat(opts.savedir,'/dbn.mat'), 'dbn');
            
            f5=figure();
            set(gcf,'Visible', 'off');
            plot(rbm.beta);
            ylim([0,1.5*rbm.max_beta])
            ylabel('beta');
            xlabel('epoch');
            grid on;
            saveas(f5,strcat(opts.savedir,'/layer',num2str(layer),'_beta.png'));
            
            
            f6=figure();
            set(gcf,'Visible', 'off');
            plot(rbm.mNZR);
            ylabel('nzr');
            xlabel('epoch');
            ylim([0,1]);
            grid on;
            title(sprintf("%.3f",mean(mNZR)))
            saveas(f6,strcat(opts.savedir,'/layer',num2str(layer),'_nzr.png'));
            
            f7=figure();
            set(gcf,'Visible', 'off');
            plot(rbm.mHoyer);
            ylabel('hsp');
            xlabel('epoch');
            ylim([0,1]);
            grid on;
            title(sprintf("%.3f",mean(mHoyer)))
            saveas(f7,strcat(opts.savedir,'/layer',num2str(layer),'_hsp.png'));
            
            
            f8=figure();
            set(gcf,'Visible', 'off');
            plot(rbm.error);
            legend({'Reconstruction error'});
            xlabel('epoch');
            grid on;
            saveas(f8,strcat(opts.savedir,'/layer',num2str(layer),'_error.png'));
            
            close all
        end  
%         figure(1); hist(rbm.W(1,:),100);          
    end
end
