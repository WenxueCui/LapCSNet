function train_LapCSN(ratio, depth, gpu)
% -------------------------------------------------------------------------
%   Description:
%       Script to train LapCSN from scratch
%
%   Input:
%       - ratio : CS sample ratio
%       - depth : numbers of conv layers in each pyramid level
%       - gpu   : GPU ID, 0 for CPU mode
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


    %% initialize opts
    opts = init_opts_cs(ratio, depth, gpu);

    %% save opts
    filename = fullfile(opts.train.expDir, 'opts.mat');
    fprintf('Save parameter %s\n', filename);
    save(filename, 'opts');

    %% setup paths
    addpath(genpath('utils'));
    addpath(fullfile(pwd, 'matconvnet/matlab'));
    vl_setupnn;

    %% initialize network
    fprintf('Initialize network...\n');
    model_filename = fullfile(opts.train.expDir, 'net-epoch-0.mat');

    fprintf(opts.train.expDir);
    
    if( ~exist(model_filename, 'file') )
        model = init_LapCSN(opts);
        fprintf('Save %s\n', model_filename);
        net = model.saveobj();
        save(model_filename, 'net');
    else
        fprintf('Load %s\n', model_filename);
        model = load(model_filename);
        model = dagnn.DagNN.loadobj(model.net);
    end

    %% load imdb
    imdb_filename = fullfile('imdb', sprintf('imdb_%s.mat', opts.data_name));
    if( ~exist(imdb_filename, 'file') )
        make_imdb(imdb_filename, opts);
    end
    fprintf('Load data %s\n', imdb_filename);
    imdb = load(imdb_filename);

    fprintf('Pre-load all images...\n');
    imdb.images.img = batch_imread(imdb.images.filename);

    %% training
    get_batch = @(x,y,mode) getBatch_LapCSN(opts,x,y,mode);

    [net, info] = vllab_cnn_train_dag(model, imdb, get_batch, opts.train, ...
                                      'val', find(imdb.images.set == 2));

