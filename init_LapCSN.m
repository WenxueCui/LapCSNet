function net = init_LapCSN(opts)
% -------------------------------------------------------------------------
%   Description:
%       create initial LapCSN model
%
%   Input:
%       - opts  : options generated from init_opts()
%
%   Output:
%       - net   : dagnn model
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

    %% parameters
    rng('default');
    rng(0) ;
    
    
    f       = opts.conv_f;
    n       = opts.conv_n;
    pad     = floor(f/2);
    depth   = opts.depth;
    scale   = opts.scale;
    level   = ceil(log(scale) / log(2));
    if( f == 3 )
        crop = [0, 1, 0, 1];
    elseif( f == 5 )
        crop = [1, 2, 1, 2];
    else
        error('Need to specify crop in deconvolution for f = %d\n', f);
    end
   
    
    net = dagnn.DagNN;
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% compassion branch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    %sigma   = opts.init_sigma;
    sigma   = sqrt( 2 / (32 * 32 * floor(32*32*opts.ratio)));
    filters = sigma * randn(32, 32, 1, floor(32*32*opts.ratio), 'single');
    biases  = [];
    
    % conv
    inputs  = { 'Input_LR' };
    outputs = { 'input_conv_compression' };
    params  = { 'input_conv_compression_f', 'input_conv_compression_b' };
    
    net.addLayer(outputs{1}, ...
                 dagnn.Conv('size', size(filters), ...
                            'pad', 0, ...
                            'stride', 32), ...
                 inputs, outputs, params);

    idx = net.getParamIndex(params{1});
    net.params(idx).value         = filters;
    net.params(idx).learningRate  = 1;
    net.params(idx).weightDecay   = 1;

    idx = net.getParamIndex(params{2});
    net.params(idx).value         = biases;
    net.params(idx).learningRate  = 0.1;
    net.params(idx).weightDecay   = 1;
    
    
    sigma   = sqrt( 2 / (opts.filter_size * opts.filter_size * floor(32*32*opts.ratio)*opts.output1));
    filters = sigma * randn(opts.filter_size, opts.filter_size, floor(32*32*opts.ratio), opts.output1, 'single');
    biases  = zeros(1, opts.output1, 'single');
    
    next_input = outputs{1};
    % conv
    inputs  = { next_input };
    outputs = { 'input_conv_compression_first' };
    params  = { 'input_conv_compression_first_f', 'input_conv_compression_first_b' };
    
    net.addLayer(outputs{1}, ...
                 dagnn.Conv('size', size(filters), ...
                            'pad', floor(opts.filter_size/2), ...
                            'stride', 1), ...
                 inputs, outputs, params);

    idx = net.getParamIndex(params{1});
    net.params(idx).value         = filters;
    net.params(idx).learningRate  = 1;
    net.params(idx).weightDecay   = 1;

    idx = net.getParamIndex(params{2});
    net.params(idx).value         = biases;
    net.params(idx).learningRate  = 0.1;
    net.params(idx).weightDecay   = 1;
    
    % ReLU
    inputs  = { 'input_conv_compression_first' };
    outputs = { 'input_relu_compression_first' };
    
    net.addLayer(outputs{1}, ...
                 dagnn.ReLU('leak', 0), ...
                 inputs, outputs);
    
    next_input = outputs{1};
    
    % reshape layer
    inputs  = { next_input };
    outputs = { 'input_reshape_compression' };
    
    net.addLayer(outputs{1}, ...
                 dagnn.vllab_dag_reshape('dims', [sqrt(opts.output1),sqrt(opts.output1)]), ...
                 inputs, outputs);
    
    next_input = outputs{1};
    
    % first reconstruction
    sigma   = sqrt( 2 / (f * f * n));
    filters = sigma * randn(f, f, 1, n, 'single');
    biases  = zeros(1, n, 'single');
    
    % conv
    inputs  = { next_input };
    outputs = { sprintf('level_conv_first_reconstruction') };
    params  = { sprintf('level_conv_first_reconstruction_f'), ...
        sprintf('level_conv_first_reconstruction_b')};
    
    net.addLayer(outputs{1}, ...
        dagnn.Conv('size', size(filters), ...
        'pad', pad, ...
        'stride', 1), ...
        inputs, outputs, params);
    
    idx = net.getParamIndex(params{1});
    net.params(idx).value         = filters;
    net.params(idx).learningRate  = 1;
    net.params(idx).weightDecay   = 1;
    
    idx = net.getParamIndex(params{2});
    net.params(idx).value         = biases;
    net.params(idx).learningRate  = 0.1;
    net.params(idx).weightDecay   = 1;
    
    % ReLU
    inputs  = { sprintf('level_conv_first_reconstruction') };
    outputs = { sprintf('level_relu_first_reconstruction') };
    
    net.addLayer(outputs{1}, ...
        dagnn.ReLU('leak', 0), ...
        inputs, outputs);
    
    next_input = outputs{1};
    
    sigma   = sqrt( 2 / (f * f * n * n));
    for d = 1: opts.deep
        filters = sigma * randn(f, f, n, n, 'single');
        biases  = zeros(1, n, 'single');
        
        % conv
        inputs  = { next_input };
        outputs = { sprintf('level_conv%d_first_reconstruction', d) };
        params  = { sprintf('level_conv%d_first_reconstruction_f', d), ...
            sprintf('level_conv%d_first_reconstruction_b', d)};
        
        net.addLayer(outputs{1}, ...
            dagnn.Conv('size', size(filters), ...
            'pad', pad, ...
            'stride', 1), ...
            inputs, outputs, params);
        
        idx = net.getParamIndex(params{1});
        net.params(idx).value         = filters;
        net.params(idx).learningRate  = 1;
        net.params(idx).weightDecay   = 1;
        
        idx = net.getParamIndex(params{2});
        net.params(idx).value         = biases;
        net.params(idx).learningRate  = 0.1;
        net.params(idx).weightDecay   = 1;
        
        % ReLU
        inputs  = { sprintf('level_conv%d_first_reconstruction', d) };
        outputs = { sprintf('level_relu%d_first_reconstruction', d) };
        
        net.addLayer(outputs{1}, ...
            dagnn.ReLU('leak', 0), ...
            inputs, outputs);
        
        next_input = outputs{1};
    end
    
    % first reconstruction last layer
    sigma   = sqrt( 2 / (f * f * n));
    filters = sigma * randn(f, f, n, 1, 'single');
    biases  = zeros(1, 1, 'single');
    
    % conv
    inputs  = { next_input };
    outputs = { sprintf('level_conv%d_first_reconstruction_last', d) };
    params  = { sprintf('level_conv%d_first_reconstruction_last_f', d), ...
        sprintf('level_conv%d_first_reconstruction_last_b', d)};
    
    net.addLayer(outputs{1}, ...
        dagnn.Conv('size', size(filters), ...
        'pad', pad, ...
        'stride', 1), ...
        inputs, outputs, params);
    
    idx = net.getParamIndex(params{1});
    net.params(idx).value         = filters;
    net.params(idx).learningRate  = 1;
    net.params(idx).weightDecay   = 1;
    
    idx = net.getParamIndex(params{2});
    net.params(idx).value         = biases;
    net.params(idx).learningRate  = 0.1;
    net.params(idx).weightDecay   = 1;
    
    next_input = outputs{1};
    next_input_first_reconstruction = next_input;
    
    inputs  = { next_input, ... 
        sprintf('level%d_HR', 0) };
    outputs = { sprintf('level%d_%s_loss', 0, opts.loss_L2) };
    
    net.addLayer(outputs{1}, ...
        dagnn.vllab_dag_loss(...
        'loss_type', opts.loss_L2), ...
        inputs, outputs);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Feature extraction branch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    sigma   = sqrt( 2 / (f * f * n));
    filters = sigma * randn(f, f, 1, n, 'single');
    biases  = zeros(1, n, 'single');
    
    % conv
    inputs  = { next_input };
    outputs = { 'input_conv' };
    params  = { 'input_conv_f', 'input_conv_b' };
    
    net.addLayer(outputs{1}, ...
                 dagnn.Conv('size', size(filters), ...
                            'pad', pad, ...
                            'stride', 1), ...
                 inputs, outputs, params);

    idx = net.getParamIndex(params{1});
    net.params(idx).value         = filters;
    net.params(idx).learningRate  = 1;
    net.params(idx).weightDecay   = 1;

    idx = net.getParamIndex(params{2});
    net.params(idx).value         = biases;
    net.params(idx).learningRate  = 0.1;
    net.params(idx).weightDecay   = 1;
    
    % ReLU
    inputs  = { 'input_conv' };
    outputs = { 'input_relu' };
    
    net.addLayer(outputs{1}, ...
                 dagnn.ReLU('leak', 0.2), ...
                 inputs, outputs);
    
    next_input = outputs{1};
    
    %% deep conv layers (f x f x n x n)    
    sigma   = sqrt( 2 / (f * f * n * n) );
    
    for s = 1 : -1 : 1
        
        % conv layers (f x f x n x n)
        for d = 1:depth
            
            filters = sigma * randn(f, f, n, n, 'single');
            biases  = zeros(1, n, 'single');

            % conv
            inputs  = { next_input };
            outputs = { sprintf('level%d_conv%d', s, d) };
            params  = { sprintf('level%d_conv%d_f', s, d), ...
                        sprintf('level%d_conv%d_b', s, d)};

            net.addLayer(outputs{1}, ...
                         dagnn.Conv('size', size(filters), ...
                                    'pad', pad, ...
                                    'stride', 1), ...
                         inputs, outputs, params);

            idx = net.getParamIndex(params{1});
            net.params(idx).value         = filters;
            net.params(idx).learningRate  = 1;
            net.params(idx).weightDecay   = 1;

            idx = net.getParamIndex(params{2});
            net.params(idx).value         = biases;
            net.params(idx).learningRate  = 0.1;
            net.params(idx).weightDecay   = 1;
			
			next_input = outputs{1};
			
            if(opts.hasBN)
			%Bnorm
			   inputs = {next_input};
			   outputs = {sprintf('level%d_bnorm%d', s, d) };
			   net.addLayer(outputs{1}, ...
			             dagnn.BatchNorm('numChannels',n, 'epsilon',1e-5), ...
						 inputs, outputs, {sprintf('gg%d',d),sprintf('bb%d',d),sprintf('mm%d',d)});
						 
			   next_input = outputs{1};
            end
			

            % ReLU
            inputs  = { next_input };
            outputs = { sprintf('level%d_relu%d', s, d) };

            net.addLayer(outputs{1}, ...
                         dagnn.ReLU('leak', 0.2), ...
                     inputs, outputs);
                 
            next_input = outputs{1};
            
        end
        
        
        %% residual prediction layer (f x f x n x 1)
        sigma   = sqrt(2 / (f * f * n));
        filters = sigma * randn(f, f, n, 1, 'single');
        biases  = zeros(1, 1, 'single');
        
        inputs  = { next_input };
        outputs = { sprintf('level%d_residual', s) };
        params  = { sprintf('level%d_residual_conv_f', s), ...
                    sprintf('level%d_residual_conv_b', s) };
        
        net.addLayer(outputs{1}, ...
            dagnn.Conv('size', size(filters), ...
                       'pad', pad, ...
                       'stride', 1), ...
            inputs, outputs, params);
        
        idx = net.getParamIndex(params{1});
        net.params(idx).value         = filters;
        net.params(idx).learningRate  = 1;
        net.params(idx).weightDecay   = 1;
        
        idx = net.getParamIndex(params{2});
        net.params(idx).value         = biases;
        net.params(idx).learningRate  = 0.1;
        net.params(idx).weightDecay   = 1;
        
        
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Image reconstruction branch
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    next_input = next_input_first_reconstruction;
    
    for s = 1 : -1 : 1
        

        
        %% residual addition layer
        inputs  = { next_input, ...
                    sprintf('level%d_residual', s) };
        outputs = { sprintf('level%d_output', s) };
        net.addLayer(outputs{1}, ...
            dagnn.Sum(), ...
            inputs, outputs);
        
        next_input = outputs{1};
        
        %% Loss layer
        inputs  = { next_input, ...
                    sprintf('level%d_HR', s) };
        outputs = { sprintf('level%d_%s_loss', s, opts.loss_L1) };
        
        net.addLayer(outputs{1}, ...
                 dagnn.vllab_dag_loss(...
                    'loss_type', opts.loss_L1), ...
                 inputs, outputs);
                
    end   
    if(opts.hasBN)         
       net.initParams();
    end
end
