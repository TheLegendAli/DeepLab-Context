clear all; close all;

% change values here
is_server = 1;
if is_server
    VOC_root_folder = '/rmt/data/pascal/VOCdevkit';
else
    VOC_root_folder = '~/dataset/PASCAL/VOCdevkit';
end

%output_mat_folder = '/rmt/work/deeplabel/exper/voc12/features/vgg128/trainval_aug/fc8_crop';
output_mat_folder = '/rmt/work/deeplabel/exper/voc12/features/vgg128_noup/val/fc8';

id = 'comp6';

%trainset = 'trainval_aug';
trainset = 'train_aug';

%testset = 'trainval_aug';
testset = 'val';

save_root_folder = output_mat_folder;


% You do not need to chage values below
seg_res_dir = [save_root_folder '/results/VOC2012/'];
save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset '_cls']);

seg_root = fullfile(VOC_root_folder, 'VOC2012');

if ~exist(save_result_folder, 'dir')
    mkdir(save_result_folder);
end

VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset);

% crop the results
load('pascal_seg_colormap.mat');

output_dir = dir(fullfile(output_mat_folder, '*.mat'));

matlabpool('4');
parfor i = 1 : numel(output_dir)
%for i = 1 : numel(output_dir)
    fprintf(1, 'processing %d (%d)...\n', i, numel(output_dir));
    
    data = load(fullfile(output_mat_folder, output_dir(i).name));
    raw_result = data.data;
    raw_result = permute(raw_result, [2 1 3]);

    img_fn = output_dir(i).name(1:end-4);
    img_fn = strrep(img_fn, '_blob_0', '');
    img = imread(fullfile(VOC_root_folder, 'VOC2012', 'JPEGImages', [img_fn, '.jpg']));
    img_row = size(img, 1);
    img_col = size(img, 2);
    
    result = raw_result(1:img_row, 1:img_col, :);
    %result = raw_result;
    [~, result] = max(result, [], 3);
    result = uint8(result) - 1;
    imwrite(result, colormap, fullfile(save_result_folder, [img_fn, '.png']));
end
matlabpool('close');

% get iou score
[accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);

    
    

