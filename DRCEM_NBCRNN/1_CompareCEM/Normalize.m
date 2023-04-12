function normalized_img = Normalize(img)
    img = double(img);
    [len,wid,band] = size(img);
    for i = 1:band
        max_val = max(max(img(:,:,i)));
        min_val = min(min(img(:,:,i)));
        for j = 1:len
            for k = 1:wid
                img(j,k,i) = (img(j,k,i)-min_val)/(max_val-min_val);
            end
        end
    end
    normalized_img = img;
end