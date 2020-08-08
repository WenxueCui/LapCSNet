function lr_all = learning_rate_policy(init_lr, step, drop, min_lr, num_epochs)
% -------------------------------------------------------------------------
%   Description:
%       function to generate a set of learning rates for each epoch
%
%   Input:
%       - init_lr   : initial learning rate
%       - step      : number of epochs to drop learning rate
%       - drop      : learning rate drop ratio
%       - min_lr    : minimum learning rate
%       - num_epochs: total number of epochs
%
%   Output:
%       - lr_all    : learning rate for epochs
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

    if( drop == 0 )
        lr_all = repmat(init_lr, 1, num_epochs);
    else
        num_drop = round(num_epochs / step) - 1;
        lr_all = init_lr * drop.^(0:num_drop);
        lr_all = repmat(lr_all, step, 1);
        lr_all = lr_all(:);
    end
    
    lr_all = max(lr_all, min_lr);
end