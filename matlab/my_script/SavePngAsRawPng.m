clear all; close all;

dataset = 'VOC2012';

orig_folder = fullfile('..', dataset, 'SegmentationClassAug_Visualization');
imgs_dir = dir(fullfile(orig_folder, '*.png'));
save_folder = ['../', dataset, '/SegmentationClassAug'];

if ~exist(save_folder, 'dir')
    mkdir(save_folder)
end

for i = 1 : numel(imgs_dir)
    fprintf(1, 'processing %d (%d) ...\n', i, numel(imgs_dir));
    
    img = imread(fullfile(orig_folder, imgs_dir(i).name));
    
    imwrite(img, fullfile(save_folder, imgs_dir(i).name));
end