addpath /users/hailey/code/03_code/HC_rfmri/RBM_DBN ... 
    /data2/alumni/khc/code/rfMRI/bakcup/MATLAB/project/ANN/tests ...
    /data2/alumni/khc/code/rfMRI/bakcup/MATLAB/project/ANN/data ...
    /data2/alumni/khc/code/rfMRI/bakcup/MATLAB/project/ANN/util...
    /data2/alumni/khc/code/rfMRI/bakcup/MATLAB/project/ANN/NN...
    /data2/alumni/khc/code/rfMRI/bakcup/MATLAB/project/ANN/SAE
clc; clear; close all;

gpuDevice(1);
thsp = [0.9]; % Target sparsity setting
mbeta = [0.01]; % Maximum beta value setting

% Display time
disp([datestr(now, 'HH:MM:SS')]);
load /data4/open_data/HCP/zscored_837.mat;
disp([datestr(now, 'HH:MM:SS')]);

%% training

hid_nodes = [5000]; % Number of hidden nodes of RBM

j = 1;
            
start = clock;
dbn.sizes       = hid_nodes;
opts.numepochs	= 15; % Number of epochs to train
opts.batchsize	= 512; % Batch size

opts.alpha      = [0.0005];
opts.momentum	= 0.1;
opts.gbrbm      = 1;
opts.max_beta   = [mbeta(j)];
opts.hsparsityTarget    = 0;
%             opts.wsparsityTarget    = [nzr(j) nzr(j) nzr(j) 0 0];
opts.wsparsityTarget    = [0 0 0 0 0];
%             opts.hoyerTarget        = [tnzr(j) tnzr(j) tnzr(j) 0 0];
%             opts.hoyerTarget        = [0.9 0.8 0.7 0 0];
opts.hoyerTarget        = [thsp(j) 0 0 0 0];
%             opts.hoyerTarget        = [0 0 0 0 0];
opts.weightPenaltyL1	= [0.0 0.0 0.0];
opts.weightPenaltyL2	= 0;
opts.dropoutFraction	= 0;
opts.denoiselv	= 0.7; % Denoising level: 0 0.3 0.5 0.7
opts.savedir = strcat('/users/hjd/hsp_results/Github_code/RBM/RBM_training/');
if ~exist(opts.savedir)
    mkdir(opts.savedir);
end
for tmp_ = 1:size(hid_nodes,2)
    mkdir(strcat(opts.savedir,'/layer',num2str(tmp_)));
end 
dbn = dbnsetup_rest(dbn, x, opts); % train_x
dbn = dbntrain_rest(dbn, x, opts); % train_x

