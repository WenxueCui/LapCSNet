function [psnr_c,ssim_c] = test_LapCSN(dataset, epoch)
% -------------------------------------------------------------------------
%   Description:
%       Script to test LapCSN on benchmark datasets
%       Compute PSNR, SSIM and IFC
%
%   Input:
%       - dataset       : testing dataset (Set5, Set14, BSDS100, urban100, manga109)
%       - epoch         : model epoch to test
%
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
    

    %% generate opts
    opts = load('test_opts/opts.mat');
    opts = opts.opts;
    
    %% setup paths
    addpath(genpath('utils'));
    
    input_dir = fullfile('datasets', dataset, 'GT');
    output_dir = fullfile(opts.train.expDir, sprintf('epoch_%d', epoch), ...
                          dataset);

    if( ~exist(output_dir, 'dir') )
        mkdir(output_dir);
    end
    
    %% Load model
    model_filename = fullfile(opts.train.expDir, sprintf('net-epoch-%d.mat', epoch));
    fprintf('Load %s\n', model_filename);
    
    net = load(model_filename);
    net = dagnn.DagNN.loadobj(net.net);
    net.mode = 'test' ;

    output_var = 'level1_output';
    output_index = net.getVarIndex(output_var);
    net.vars(output_index).precious = 1;

    if( opts.gpu )
        gpuDevice(opts.gpu)
        net.move('gpu');
    end

    %% load image list
    list_filename = sprintf('lists/%s.txt', dataset);
    img_list = load_list(list_filename);
    num_img = length(img_list);
    

    %% testing
    PSNR = zeros(num_img, 1);
    SSIM = zeros(num_img, 1);
    IFC  = zeros(num_img, 1);
    
    for i = 1:num_img
        
        img_name = img_list{i};
        fprintf('Process %s %d/%d: %s\n', dataset, i, num_img, img_name);
        
        % Load HR image
        input_filename = fullfile(input_dir, sprintf('%s.png', img_name));
        img_GT = im2double(imread(input_filename));
        img_GT = mod_crop(img_GT, 32);
    
        % generate LR image
        img_Input = img_GT;
            
        % apply LapSRN
        img_output = Meas_LapCSN(img_Input, net, opts);
            
        % save result
        output_filename = fullfile(output_dir, sprintf('%s.png', img_name));
        fprintf('Save %s\n', output_filename);
        imwrite(img_output, output_filename);

        %% evaluate
        img_output = im2double(im2uint8(img_output)); % quantize pixel values
        
        % convert to gray scale
        img_GT = rgb2ycbcr(img_GT); img_GT = img_GT(:, :, 1);
        img_output = rgb2ycbcr(img_output); img_output = img_output(:, :, 1);
        
        % crop boundary
        %img_GT = shave_bd(img_GT, test_scale);
        %img_HR = shave_bd(img_HR, test_scale);
        
        % evaluate
        PSNR(i) = psnr(img_GT, img_output);
        SSIM(i) = ssim(img_GT, img_output);
        
        % comment IFC to speed up testing
%         IFC(i) = ifcvec(img_GT, img_HR);
%         if( ~isreal(IFC(i)) )
%             IFC(i) = 0;
%         end

    end
    
    PSNR(end+1) = mean(PSNR);
    SSIM(end+1) = mean(SSIM);
    IFC(end+1)  = mean(IFC);
    
    fprintf('Average PSNR = %f\n', PSNR(end));
    fprintf('Average SSIM = %f\n', SSIM(end));
    fprintf('Average IFC = %f\n', IFC(end));
    
    filename = fullfile(output_dir, 'PSNR.txt');
    save_matrix(PSNR, filename);

    filename = fullfile(output_dir, 'SSIM.txt');
    save_matrix(SSIM, filename);
    
    filename = fullfile(output_dir, 'IFC.txt');
    save_matrix(IFC, filename);
    
    psnr_c = PSNR;
    ssim_c = SSIM;
