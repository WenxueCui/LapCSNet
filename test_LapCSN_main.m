function res = test_LapCSN_main(a,b)
% -------------------------------------------------------------------------
%   Description:
%       Script to test LapCSN on benchmark datasets
%
%   Input:
%       - a       : start of epoch
%       - b         : end of epoch
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
    
maxs_psnr = 0.0;
index_psnr = 0;
maxs_ssim = 0.0;
index_ssim = 0;
fid = fopen('res_mat_1_10_depth_4_set5.txt','a+');
for i = a:b
    fprintf(['processing',' ',num2str(i),' ','epoch',' ','.......']);
    fprintf('\n');
    [psnr_c,ssim_c] = test_LapCSN('Set5',i);
    
    %psnr_c = [0.0345,45.9,23.4];
    if psnr_c(end) > maxs_psnr
        maxs_psnr = psnr_c(end); 
        index_psnr = i;
    end
    if ssim_c(end) > maxs_ssim
        maxs_ssim = ssim_c(end); 
        index_ssim = i;
    end
    fprintf(fid,['epoch-- ',num2str(i),'     ',num2str(psnr_c')]);
    fprintf(fid,'\r\n');
    fprintf(fid,['epoch-- ',num2str(i),'     ',num2str(ssim_c')]);
    fprintf(fid,'\r\n');fprintf(fid,'\r\n');fprintf(fid,'\r\n');
end

res = [maxs_psnr,maxs_ssim];

fprintf(fid,'\r\n');fprintf(fid,'\r\n');fprintf(fid,'\r\n');
fprintf(fid,['index-- ',num2str(index_psnr),'     ',num2str(res(1))]);
fprintf(fid,'\r\n');fprintf(fid,'\r\n');fprintf(fid,'\r\n');
fprintf(fid,['index-- ',num2str(index_ssim),'     ',num2str(res(2))]);
fclose(fid);
end