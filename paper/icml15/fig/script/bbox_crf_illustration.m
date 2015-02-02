addpath('~/workspace/ext/export_fig');

load('pascal_seg_colormap.mat');

img_fn = '2010_002137';

jpeg_folder = '~/dataset/PASCAL/VOCdevkit/VOC2012/JPEGImages';
bbox_folder = '~/dataset/PASCAL/VOCdevkit/VOC2012/SegmentationClass';

img   = imread(fullfile(jpeg_folder, [img_fn, '.jpg']));
bbox = imread(fullfile(bbox_folder, [img_fn '.png']));

labels = unique(bbox(:));

figure(1), imshow(img), hold on;

for i = 1 : length(labels)
    label = labels(i);
    
    if label == 0 || label == 255
        continue;
    end
    
    [row col] = find(bbox == label);
    
    min_row = min(row);
    max_row = max(row);
    min_col = min(col);
    max_col = max(col);
    
    img_row = size(img, 1);
    img_col = size(img, 2);
    
    rectangle('Position', [min_col, min_row, (max_col-min_col+1), (max_row-min_row+1)], ...
        'LineWidth', 8, ...
        'EdgeColor', colormap(label+1, :));
    
end

fn = fullfile('..', 'bbox_crf_illustration.jpg');
export_fig(fn);