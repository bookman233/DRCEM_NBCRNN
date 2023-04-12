clc
clear
close all
format long

kind_num = [1,2,7];
minRect = 64;
thresh = 0.75;
RGB_Band = 50;
addpath('./Hyper_data');

gt = load('Salinas_gt.mat');
hyper_img = load('Salinas_corrected.mat');
gt_data = gt.salinas_gt;
hyper_img_data = im2double(Normalize(hyper_img.salinas_corrected(:,:,:)));

[len, wid, band] = size(hyper_img_data);
hyper_img_reshape = reshape(hyper_img_data, len*wid, band);
img_GT_data = get_GT(gt_data, kind_num);

%% Calculate the correlation matrix
R=zeros(band);
R = hyper_img_reshape'*hyper_img_reshape;
R = (R)/(len*wid);

%% Generate prior information matrix D
d_data = zeros(band, length(kind_num));
for index = 1:length(kind_num)
    d_data(:, index) = getd_original(kind_num(index), gt_data, hyper_img_data);
end
D = d_data;

%% MTICEM
In_H = R;
In_f = zeros(band, 1);
In_A = -D';
In_b = -ones(3,1);
In_x = quadprog(In_H,In_f,In_A,In_b);

%% MTCEM
En_H = R;
En_f = zeros(band, 1);
Aeq = D';
beq = ones(3,1);
En_x = quadprog(En_H,En_f,[],[],Aeq,beq);

%% Generate Figure
In_y = hyper_img_reshape*In_x;
In_Y = reshape(In_y,len,wid);

En_y = hyper_img_reshape*En_x;
En_Y = reshape(En_y,len,wid);

figure
imshow(In_Y,'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,wid,len]);

figure
imshow(En_Y,'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,wid,len]);

figure
level=graythresh(In_Y);
Bin_In_Y=imbinarize(In_Y,level);
imshow(Bin_In_Y);
title('MTICEM')

figure
level=graythresh(En_Y);
Bin_En_Y=imbinarize(En_Y,level);
imshow(Bin_En_Y);
title('MTCEM')

[Mean_In, SD_In, MP_In, OA_In, AA_In, PA_In, kappa_In] = Index_AUC(img_GT_data, Bin_In_Y);
[Mean_En, SD_En, MP_En, OA_En, AA_En, PA_En, kappa_En] = Index_AUC(img_GT_data, Bin_En_Y);




