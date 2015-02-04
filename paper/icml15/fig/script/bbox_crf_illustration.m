clear all; close all;

addpath('~/workspace/ext/export_fig');

load('pascal_seg_colormap.mat');

img_fn = '2009_002382';

jpeg_folder = '~/dataset/PASCAL/VOCdevkit/VOC2012/JPEGImages';
seg_cls_folder = '~/dataset/PASCAL/VOCdevkit/VOC2012/SegmentationClass';
seg_ins_folder = '~/dataset/PASCAL/VOCdevkit/VOC2012/SegmentationObject';

img   = imread(fullfile(jpeg_folder, [img_fn, '.jpg']));
seg_cls = imread(fullfile(seg_cls_folder, [img_fn '.png']));
seg_ins = imread(fullfile(seg_ins_folder, [img_fn '.png']));

[inst_label, ~, ~] = unique([seg_ins(:), seg_cls(:)], 'rows');

figure(1), imshow(img), hold on;

for i = 1 : size(inst_label, 1)
    label = inst_label(i, 2);
    
    if label == 0 || label == 255
        continue;
    end
    
    [row col] = find(seg_ins == i-1);
    
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

fn = fullfile('..', [img_fn '.jpg']);
export_fig(fn);