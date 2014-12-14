% change values here
is_server = 1;
if is_server
    VOC_root_folder = '/rmt/data/pascal/VOCdevkit';
else
    VOC_root_folder = '~/dataset/PASCAL/VOCdevkit';
end

output_mat_folder = 'xxx';

id = 'comp6';
trainset = 'train';
testset = 'val';

save_root_folder = '/rmt/work/deeplabel/exper/voc12';

% You do not need to chage values below
save_result_folder = fullfile(save_root_folder, 'results', 'VOC2012', 'Segmentation', [id '_' testset '_cls']);
if ~exist(save_result_folder, 'dir')
    mkdir(save_result_folder);
end

VOCopts = GetVOCopts(trainset, testset);

% crop the results
load('pascal_seg_colormap.mat');

output_dir = dir(output_mat_folder, '*.mat');

for i = 1 : numel(output_dir)
    data = load(fullfile(output_mat_folder, output_dir(i).name));
    raw_result = data.raw_result;
    
    img_fn = output_dir(i).name(1:end-4);
    img = imread(fullfile(VOC_root_folder, 'VOC2012', 'JPEGImages', [img_fn, '.jpg']));
    img_row = size(img, 1);
    img_col   = size(img, 2);
    
    result = raw_result(1:img_row, 1:img_col, :);
    imwrite(result, colormap, fullfile(save_result_folder, [img_fn, '.png']));
end

% get iou score
[accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);

    
    

