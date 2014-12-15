data_im = reshape(1:8, [2 2 2]); 
w_im = reshape(0.1:0.1:1.8, [3, 3, 2]);
height = size(data_im, 1);
width = size(data_im, 2);
channels = size(data_im, 3);
stride_h = 2;
stride_w = 2;
kernel_h = size(w_im, 1);
kernel_w = size(w_im, 2);
pad_h = 0;
pad_w = 0;

height_col = (height - 1) * stride_h + kernel_h - 1 * pad_h;
width_col = (width - 1) * stride_w + kernel_w - 1 * pad_w;
channels_col = channels * kernel_w * kernel_h;

data_col = zeros(channels_col, width_col*height_col);

for c = 0 : channels_col - 1
    w_offset = mod(c, kernel_w);
    h_offset = mod(floor( c / kernel_w), kernel_h);
    c_im = floor(c / kernel_h / kernel_w);
    
    for h = 0 : height - 1
        for w = 0 : width - 1
            h_pad = h * stride_h - pad_h + h_offset;
            w_pad = w * stride_w - pad_w + w_offset;
            
            if (h_pad >= 0 && h_pad < height_col && w_pad >= 0 && w_pad < width_col)
                data_col(c+1, w_pad * height_col + h_pad + 1) =  data_im(w+1, h+1, c_im+1);
            else
                data_col(c+1, w_pad * height_col + h_pad + 1) =  0;
            end
            %data_col[ (c * height_col + h_pad) * width_col + w_pad] = data_im[ (c_im * height + h) * width + w];
        end
    end
end
         
%col2im
data = zeros(size(data_im));

for c = 0 : channels_col - 1
    w_offset = mod(c, kernel_w);
    h_offset = mod(floor( c / kernel_w), kernel_h);
    c_im = floor(c / kernel_h / kernel_w);
    
    for h = 0 : height - 1
        for w = 0 : width - 1
            h_pad = h * stride_h - pad_h + h_offset;
            w_pad = w * stride_w - pad_w + w_offset;
            
            if (h_pad >= 0 && h_pad < height_col && w_pad >= 0 && w_pad < width_col)
                data(w+1, h+1, c_im+1) = data(w+1, h+1, c_im+1) + data_col(c+1, w_pad * height_col + h_pad + 1);
            end
            %data_col[ (c * height_col + h_pad) * width_col + w_pad] = data_im[ (c_im * height + h) * width + w];
        end
    end
end