function opts = init_opts_cs(ratio, depth, gpu)
% -------------------------------------------------------------------------
%   Description:
%       Generate all options for LapCSN
%
%   Input:
%       - ratio : the subrate of CS sampling.
%       - depth : number of conv layers in one pyramid level
%       - gpu   : GPU ID, 0 for CPU mode
%
%   Output:
%       - opts  : all options for LapCSN
%

%   Citation: 
%       An efficient deep convolutional laplacian pyramid architecture for CS reconstruction at low sampling ratios
%       Wenxue Cui, Heyao Xu, Xinwei Gao, Shengping Zhang, Feng Jiang, Debin Zhao
%       IEEE International Conference on Acoustics, Speech and Signal
%       Processing (ICASSP), 2018
%
%   Contact:
%       WenxueCui
%       wenxuecui@stu.hit.edu.cn
%       Harbin Institute of Technology, China
% -------------------------------------------------------------------------

    %% bellow is the my own vars.
    opts.deep = 3; % first reconstruction layers.
    opts.ratio = ratio;  % the sample ratio.
    opts.output1 = 64;  % used in reshape.
    opts.filter_size = 1;  % reconstruction filter. 
    opts.hasBN = 0;
    
    if ((1/opts.ratio)>=64)
        opts.output1 = 1024;
        opts.scale = 16;
    elseif((1/opts.ratio)>= 16)
        opts.output1 = 1024;
        opts.scale = 8;
    elseif((1/opts.ratio)>= 4)
        opts.output1 = 1024;
        opts.scale = 4;
    end
    %% network options
    opts.depth              = depth;
    opts.weight_decay       = 0.0001;
    opts.init_sigma         = 0.001;
    opts.conv_f             = 3;
    opts.conv_n             = 64;
    opts.loss_L1            = 'L1';
	opts.loss_L2            = 'L2';
    opts.loss               = 'L1_L2';
    opts.cs                 = 2;

    %% training options
    opts.gpu                = gpu;
    opts.batch_size         = 64;
    opts.num_train_batch    = 1000;     % number of training batch in one epoch
    opts.num_valid_batch    = 100;      % number of validation batch in one epoch
    opts.lr                 = 1e-5;     % initial learning rate
    opts.lr_step            = 50;       % number of epochs to drop learning rate
    opts.lr_drop            = 0.5;      % learning rate drop ratio
    opts.lr_min             = 1e-6;     % minimum learning rate
    opts.patch_size         = 128;
    opts.data_augmentation  = 1;

    %% dataset options
    opts.train_dataset          = {};
    opts.train_dataset{end+1}   = 'T91';
    opts.train_dataset{end+1}   = 'BSDS200';
    %opts.train_dataset{end+1}   = 'General100';
    opts.valid_dataset          = {};
    opts.valid_dataset{end+1}   = 'Set5';
    opts.valid_dataset{end+1}   = 'Set14';
    opts.valid_dataset{end+1}   = 'BSDS100';


    %% setup model name
    opts.data_name = 'train';
    for i = 1:length(opts.train_dataset)
        opts.data_name = sprintf('%s_%s', opts.data_name, opts.train_dataset{i});
    end

    opts.net_name = sprintf('LapSRN_x%3f_depth%d_%s_cs%d', ...
                            opts.ratio, opts.depth, opts.loss,opts.cs);

    opts.model_name = sprintf('%s_%s_pw%d_lr%s_step%d_drop%s_min%s', ...
                            opts.net_name, ...
                            opts.data_name, opts.patch_size, ...
                            num2str(opts.lr), opts.lr_step, ...
                            num2str(opts.lr_drop), num2str(opts.lr_min));


    %% setup dagnn training parameters
    if( opts.gpu == 0 )
        opts.train.gpus     = [];
    else
        opts.train.gpus     = [opts.gpu];
    end
    opts.train.batchSize    = opts.batch_size;
    opts.train.numEpochs    = 1000;
    opts.train.continue     = true;
    opts.train.learningRate = learning_rate_policy(opts.lr, opts.lr_step, opts.lr_drop, ...
                                                   opts.lr_min, opts.train.numEpochs);

    opts.train.expDir = fullfile('models', opts.model_name) ; % model output dir
    if( ~exist(opts.train.expDir, 'dir') )
        mkdir(opts.train.expDir);
    end

    opts.train.model_name       = opts.model_name;
    opts.train.num_train_batch  = opts.num_train_batch;
    opts.train.num_valid_batch  = opts.num_valid_batch;
    
    % setup loss
    opts.level = ceil(log(opts.scale) / log(2));
    opts.train.derOutputs = {};
    for s = 1 : -1 : 0
        if(s == 1)
           opts.train.derOutputs{end+1} = sprintf('level%d_%s_loss', s, opts.loss_L1);
           opts.train.derOutputs{end+1} = 1;
        else
           opts.train.derOutputs{end+1} = sprintf('level%d_%s_loss', s, opts.loss_L2);
           opts.train.derOutputs{end+1} = 1;
        end
    end


end