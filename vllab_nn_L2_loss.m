function Y = vllab_nn_L2_loss(X, Z, dzdy)
% -------------------------------------------------------------------------
%   Description:
%       L2 (MSE) loss function used in MatConvNet NN
%       forward : Y = vllab_nn_L2_loss(X, Z)
%       backward: Y = vllab_nn_L2_loss(X, Z, dzdy)
%
%   Input:
%       - X     : predicted data
%       - Z     : ground truth data
%       - dzdy  : the derivative of the output
%
%   Output:
%       - Y     : loss when forward, derivative of loss when backward
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

    if nargin <= 2
        diff = (X - Z) .^ 2;
        Y = 0.5 * sum(diff(:));
    else
        Y = (X - Z) * dzdy;
    end
end
