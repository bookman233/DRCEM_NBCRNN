clc
clear
close all

format long
addpath('./Hyper_data');

kind_num = [1,2,7];
minRect = 64;
thresh = 0.75;

gt = load('Salinas_gt.mat');
hyper_img = load('Salinas_corrected.mat');
gt_data = gt.salinas_gt;

count = 1;
band_sequence = 110:5:200;
min_band = min(band_sequence);
data_store = zeros(7,length(band_sequence));
En_data_store = zeros(7,length(band_sequence));
In_data_store = zeros(7,length(band_sequence));

for index_iter = band_sequence
    hyper_img_data = im2double(Normalize(hyper_img.salinas_corrected(:,:,(min_band-4):index_iter)));
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

    %% NBCRNN for solving 
    init_1 = ones(band,1);
    init_2 = zeros(band,1);
    init_3 = zeros(1);
    x_init = [init_1; init_2; init_3];

    timeSpan = 0: 0.001: 20;
    options = odeset();
    [t, x] = ode45(@Core_NTZNN, timeSpan, x_init, options, R, D);
    solution = x(end,1:band)';

    y = hyper_img_reshape*solution;
    Y = reshape(y,len,wid);

    level=graythresh(Y);
    Bin_Y=imbinarize(Y,level);
    
    %% Matlab QP solver
    En_H = R;
    En_f = zeros(band, 1);
    Aeq = D';
    beq = ones(length(kind_num),1);
    En_x = quadprog(En_H,En_f,[],[],Aeq,beq);

    In_H = R;
    In_f = zeros(band, 1);
    In_A = -D';
    In_b = -ones(length(kind_num),1);
    In_x = quadprog(In_H,In_f,In_A,In_b);

    %% Generate image
    En_y = hyper_img_reshape*En_x;
    En_Y = reshape(En_y,len,wid);
    In_y = hyper_img_reshape*In_x;
    In_Y = reshape(In_y,len,wid);

    % Binary image of MTCEM
    level=graythresh(En_Y);
    Bin_En_Y=imbinarize(En_Y,level);
    % Binary image of MTICEM
    level=graythresh(In_Y);
    Bin_In_Y=imbinarize(In_Y,level);
    % =====================================================

    [Mean, SD, MP, OA, AA, PA, kappa] = Index_AUC(img_GT_data, Bin_Y);
    [En_Mean, En_SD, En_MP, En_OA, En_AA, En_PA, En_kappa] = Index_AUC(img_GT_data, Bin_En_Y);
    [In_Mean, In_SD, In_MP, In_OA, In_AA, In_PA, In_kappa] = Index_AUC(img_GT_data, Bin_In_Y);
    
    data_store(:,count) = [Mean, SD, MP, OA, AA, PA, kappa];
    En_data_store(:,count) = [En_Mean, En_SD, En_MP, En_OA, En_AA, En_PA, En_kappa];
    In_data_store(:,count) = [In_Mean, In_SD, In_MP, In_OA, In_AA, In_PA, In_kappa];
    
    count = count + 1;
    index_iter
end

figure(1)
plot(band_sequence, data_store(4,:), '-o', 'LineWidth', 2)
hold on;
plot(band_sequence, In_data_store(4,:), '-s', 'LineWidth', 2)
hold on;
plot(band_sequence, En_data_store(4,:), '-^', 'LineWidth', 2)
hold on;
xlabel('Number of Bands')
ylabel('Overall Accuracy')
legend('NIDCEM','MTICEM','MTCEM')
grid on;
hold on;

figure(2)
plot(band_sequence, data_store(5,:), '-o', 'LineWidth', 2)
hold on;
plot(band_sequence, In_data_store(5,:), '-s', 'LineWidth', 2)
hold on;
plot(band_sequence, En_data_store(5,:), '-^', 'LineWidth', 2)
hold on;
xlabel('Number of Bands')
ylabel('Average Accuracy')
legend('NIDCEM','MTICEM','MTCEM')
grid on;
hold on;

figure(3)
plot(band_sequence, data_store(6,:), '-o', 'LineWidth', 2)
hold on;
plot(band_sequence, In_data_store(6,:), '-s', 'LineWidth', 2)
hold on;
plot(band_sequence, En_data_store(6,:), '-^', 'LineWidth', 2)
hold on;
xlabel('Number of Bands')
ylabel('Accuracy of Producer')
legend('NIDCEM','MTICEM','MTCEM')
grid on;
hold on;

figure(4)
plot(band_sequence, data_store(7,:), '-o', 'LineWidth', 2)
hold on;
plot(band_sequence, In_data_store(7,:), '-s', 'LineWidth', 2)
hold on;
plot(band_sequence, En_data_store(7,:), '-^', 'LineWidth', 2)
hold on;
xlabel('Number of Bands')
ylabel('Kappa')
legend('NIDCEM','MTICEM','MTCEM')
grid on;
hold on;

