function img_HR = Meas_LapCSN(img_LR, net, opts)
% -------------------------------------------------------------------------
%   Description:
%       function to apply LapCSN
%
%   Input:
%       - img_LR: initial reconstruction.
%       - net   : LapCSN model
%       - opts  : options generated from init_opts()
%
%   Output:
%       - img_HR: Reconstructed image
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

    %% setup
    net.mode = 'test' ;
    output_var = 'level1_output';
    output_index = net.getVarIndex(output_var);
    net.vars(output_index).precious = 1;
    
    % RGB to YUV
    if( size(img_LR, 3) > 1 )
        img_LR = rgb2ycbcr(img_LR);
    end
    
    % extract Y
    y = single(img_LR(:, :, 1));
    
    if( opts.gpu )
        y = gpuArray(y);
    end
    
    % bicubic upsample UV
    img_HR = single(img_LR);
    

    % forward
    inputs = {'Input_LR', y};
    net.eval(inputs);
    y = gather(net.vars(output_index).value);
        
    % resize if size does not match the output image
    if( size(y, 1) ~= size(img_HR, 1) )
        y = imresize(y, [size(img_HR, 1), size(img_HR, 2)]);
    end
    
    img_HR(:, :, 1) = double(y);
        
    % YUV to RGB
    if( size(img_HR, 3) > 1 )
        img_HR = ycbcr2rgb(im2uint8(img_HR));
    end
        
    
end