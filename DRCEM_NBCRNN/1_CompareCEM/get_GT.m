function output = get_GT(gt_data, kinds)
    [len, wid] = size(gt_data);
    img_GT_data = zeros(len,wid);
    
    for index = 1:length(kinds)
        for i_len = 1:len
            for i_wid = 1:wid
                if gt_data(i_len, i_wid) == kinds(index)
                    img_GT_data(i_len, i_wid) = 1;
                end
            end
        end
    end
    
    output = img_GT_data;
end