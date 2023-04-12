clc
clear
close all
format long

kind_num = [1,2,7];
addpath('./Hyper_data');

gt = load('Salinas_gt.mat');
hyper_img = load('Salinas_corrected.mat');
gt_data = gt.salinas_gt;
hyper_img_data = im2double(Normalize(hyper_img.salinas_corrected(:,:,:)));
[len, wid, band] = size(hyper_img_data);
hyper_img_reshape = reshape(hyper_img_data, len*wid, band);
img_GT_data = get_GT(gt_data, kind_num);

R=zeros(band);
R = hyper_img_reshape'*hyper_img_reshape;
R = (R)/(len*wid);

d_data = zeros(band, length(kind_num));
for index = 1:length(kind_num)
    d_data(:, index) = getd_original(kind_num(index), gt_data, hyper_img_data);
end
D = d_data;

kind_local = getd_improve(kind_num(1),gt_data);
[kind_local_len, ~] = size(kind_local);

init_1 = ones(band,1);
init_2 = zeros(band,1);
init_3 = zeros(1);
x_init = [init_1; init_2; init_3];

timeSpan = 0: 0.01: 10;
options = odeset();
[t, x] = ode45(@Core_NBCRNN, timeSpan, x_init, options, R, D);
solution = x(end,1:band)';

y = hyper_img_reshape*solution;
Y = reshape(y,len,wid);

figure(1);
Normalize_Y = Normalize(Y);
imshow(Y,'border','tight','initialmagnification','fit');
set (gcf,'Position',[0,0,wid,len]);

figure(2);
level=graythresh(Y);
Bin_Y=imbinarize(Y,level);
imshow(Bin_Y);

[Mean, SD, MP, OA, AA, PA, kappa] = Index_AUC(img_GT_data, Bin_Y);

