function d = getd_original(kind_num, gt_data, hyper_img_data)

    addpath('./Hyper_data');

    kind = kind_num;
    [~,~,band] = size(hyper_img_data);
    [gt_lan, gt_wid] = size(gt_data);

    kind_pix_num = 0;
    for i = 1:gt_lan
        for j = 1:gt_wid
            if gt_data(i,j) == kind
                kind_pix_num = kind_pix_num + 1;
            end
        end
    end

    kind_local = zeros(kind_pix_num,2);
    count = 0;
    for i = 1:gt_lan
        for j = 1:gt_wid
            if gt_data(i,j) == kind
                count = count + 1;
                kind_local(count,:) = [i,j];
            end
        end
    end

    tempOne = zeros(1,1,band);
    for i = 1:kind_pix_num
        x = kind_local(i,1);
        y = kind_local(i,2);
        tempOne = tempOne + hyper_img_data(x,y,:);
    end
    tempOne = reshape(tempOne,band,1);
    tempOne = tempOne ./ kind_pix_num;
    d =tempOne;
end