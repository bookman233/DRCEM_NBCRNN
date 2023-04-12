clc
clear
close all
format long

kind_num = [1,2,7];
RGB_Band = 50;
addpath('./Hyper_data');

gt = load('Salinas_gt.mat');
hyper_img = load('Salinas_corrected.mat');
gt_data = gt.salinas_gt;
hyper_img_data = im2double(Normalize(hyper_img.salinas_corrected(:,:,:)));


timeSpan = 0: 0.01: 15;
noise_sequence = 0:0.1:5;
count = 1;
data_store = zeros(7,length(noise_sequence));
OZNN_data_store = zeros(7,length(noise_sequence));
GNN_data_store = zeros(7,length(noise_sequence));
NTGNN_data_store = zeros(7,length(noise_sequence));

for index_iter = noise_sequence
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

    %% NBCRNN
    init_1 = ones(band,1);
    init_2 = zeros(band,1);
    init_3 = zeros(1);
    x_init = [init_1; init_2; init_3];

    options = odeset();
    [NTZNN_t, x] = ode45(@Core_Noise_NBCZNN, timeSpan, x_init, options, R, D, index_iter);
    solution_NTZNN = x(end,1:band)';

    y = hyper_img_reshape*solution_NTZNN;
    NTZNN_Y = reshape(y,len,wid);
    level=graythresh(NTZNN_Y);
    NTZNN_Bin_Y=imbinarize(NTZNN_Y,level);
    
    %% RNINN
    init_1 = ones(band,1);
    init_2 = zeros(band,1);
    init_3 = zeros(1);
    x_init = [init_1; init_2; init_3];

    options = odeset();
    [NTGNN_t, x] = ode45(@Core_Noise_NTGNN, timeSpan, x_init, options, R, D, index_iter);
    solution_NTGNN = x(end,1:band)';

    y = hyper_img_reshape*solution_NTGNN;
    NTGNN_Y = reshape(y,len,wid);
    level=graythresh(NTGNN_Y);
    NTGNN_Bin_Y=imbinarize(NTGNN_Y,level);
    
    %% OZNN
    x_init_OZNN = ones(band + 1,1);
    
    options = odeset();
    [OZNN_t, OZNN_x] = ode45(@Core_Noise_OZNN, timeSpan, x_init_OZNN, options, R, D, index_iter);
    OZNN_solution = OZNN_x(end,1:(end-1))';
    
    y = hyper_img_reshape*OZNN_solution;
    OZNN_Y = reshape(y,len,wid);
    level=graythresh(OZNN_Y);
    OZNN_Bin_Y=imbinarize(OZNN_Y,level);
    
    %% GNN
    x_init_GNN = ones(band + 1,1);
    
    options = odeset();
    [GNN_t, GNN_x] = ode45(@Core_Noise_GNN, timeSpan, x_init_GNN, options, R, D, index_iter);
    GNN_solution = GNN_x(end,1:(end-1))';
    
    y = hyper_img_reshape*GNN_solution;
    GNN_Y = reshape(y,len,wid);
    level=graythresh(GNN_Y);
    GNN_Bin_Y=imbinarize(GNN_Y,level);
    
    [Mean, SD, MP, OA, AA, PA, kappa] = Index_AUC(img_GT_data, NTZNN_Bin_Y);
    [O_Mean, O_SD, O_MP, O_OA, O_AA, O_PA, O_kappa] = Index_AUC(img_GT_data, OZNN_Bin_Y);
    [G_Mean, G_SD, G_MP, G_OA, G_AA, G_PA, G_kappa] = Index_AUC(img_GT_data, GNN_Bin_Y);
    [NTG_Mean, NTG_SD, NTG_MP, NTG_OA, NTG_AA, NTG_PA, NTG_kappa] = Index_AUC(img_GT_data, NTGNN_Bin_Y);
    
    data_store(:,count) = [Mean, SD, MP, OA, AA, PA, kappa];
    OZNN_data_store(:,count) = [O_Mean, O_SD, O_MP, O_OA, O_AA, O_PA, O_kappa];
    GNN_data_store(:,count) = [G_Mean, G_SD, G_MP, G_OA, G_AA, G_PA, G_kappa];
    NTGNN_data_store(:,count) = [NTG_Mean, NTG_SD, NTG_MP, NTG_OA, NTG_AA, NTG_PA, NTG_kappa];
    
    count = count + 1;
    index_iter
end

figure(1)
plot(noise_sequence, data_store(4,:), '-o', 'LineWidth', 2)
hold on;
plot(noise_sequence, OZNN_data_store(4,:), '-s', 'LineWidth', 2)
hold on;
plot(noise_sequence, GNN_data_store(4,:), '-^', 'LineWidth', 2)
hold on;
plot(noise_sequence, NTGNN_data_store(4,:), '-^', 'LineWidth', 2)
hold on;
xlabel('Number of Bands')
ylabel('Overall Accuracy')
legend('NBCRNN','OZNN','GNN','RNINN')
grid on;
hold on;

figure(2)
plot(noise_sequence, data_store(5,:), '-o', 'LineWidth', 2)
hold on;
plot(noise_sequence, OZNN_data_store(5,:), '-s', 'LineWidth', 2)
hold on;
plot(noise_sequence, GNN_data_store(5,:), '-^', 'LineWidth', 2)
hold on;
plot(noise_sequence, NTGNN_data_store(5,:), '-^', 'LineWidth', 2)
hold on;
xlabel('Number of Bands')
ylabel('Average Accuracy')
legend('NBCRNN','OZNN','GNN','RNINN')
grid on;
hold on;

figure(3)
plot(noise_sequence, data_store(6,:), '-o', 'LineWidth', 2)
hold on;
plot(noise_sequence, OZNN_data_store(6,:), '-s', 'LineWidth', 2)
hold on;
plot(noise_sequence, GNN_data_store(6,:), '-^', 'LineWidth', 2)
hold on;
plot(noise_sequence, NTGNN_data_store(6,:), '-^', 'LineWidth', 2)
hold on;
xlabel('Number of Bands')
ylabel('Accuracy of Producer')
legend('NBCRNN','OZNN','GNN','RNINN')
grid on;
hold on;

figure(4)
plot(noise_sequence, data_store(7,:), '-o', 'LineWidth', 2)
hold on;
plot(noise_sequence, OZNN_data_store(7,:), '-s', 'LineWidth', 2)
hold on;
plot(noise_sequence, GNN_data_store(7,:), '-^', 'LineWidth', 2)
hold on;
plot(noise_sequence, NTGNN_data_store(7,:), '-^', 'LineWidth', 2)
hold on;
xlabel('Number of Bands')
ylabel('Kappa')
legend('NBCRNN','OZNN','GNN','RNINN')
grid on;
hold on;

