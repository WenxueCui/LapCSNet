function img_list = batch_imread(batch)
% -------------------------------------------------------------------------
%   Description:
%       read a batch of images
%
%   Input:
%       - batch : array of ID to fetch
%
%   Output:
%       - img_list: batch of images
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

    img_list = cell(length(batch), 1);
    
    for i = 1:length(batch)
         img = imread(batch{i});
         img = rgb2ycbcr(img);
         img = img(:, :, 1);
         
         img_list{i} = im2single(img);
    end

end