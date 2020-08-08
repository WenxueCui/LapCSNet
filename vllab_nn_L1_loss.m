function Y = vllab_nn_L1_loss(X, Z, dzdy)
% -------------------------------------------------------------------------
%   Description:
%       L1 (Charbonnier) loss function
%       forward : Y = vllab_nn_L1_loss(X, Z)
%       backward: Y = vllab_nn_L1_loss(X, Z, dzdy)
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

    eps = 1e-6;
    d = X - Z;
    e = sqrt( d.^2 + eps );
    
    if nargin <= 2
        Y = sum(e(:));
    else
        Y = d ./ e;
        Y = Y .* dzdy;
    end
    
end
