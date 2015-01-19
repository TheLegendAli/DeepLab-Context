clear all; close all;

% change values here
is_server       = 1;
is_mat          = 1;   % the results are saved as mat or png
has_postprocess = 1;   % has done densecrf post processing or not

pos_w          = 3;
pos_x_std      = 3;

bi_w      = 5;% 3;    %5;
bi_x_std  = 50;%95;   %50;
bi_r_std  = 10;%3;    %10;

id         = 'comp6';
%trainset  = 'trainval_aug';
trainset   = 'train_aug';

testset   = 'val';
%testset    = 'test';            %'val', 'test'

model_name = 'vgg128_noup_pool3';   %'vgg128_noup', 'vgg128_noup_glob', 'vgg128_ms'
feature_name = 'features';        %'features', 'features4', 'features2'

if is_server
    VOC_root_folder = '/rmt/data/pascal/VOCdevkit';
else
    VOC_root_folder = '~/dataset/PASCAL/VOCdevkit';
end

if has_postprocess
  post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std); 
else
  post_folder = 'post_none';
end

output_mat_folder = fullfile('/rmt/work/deeplabel/exper/voc12', feature_name, model_name, testset, 'crf');

if strcmp(feature_name, 'features')
  save_root_folder = fullfile('/rmt/work/deeplabel/exper/voc12/res', model_name, testset, 'crf', post_folder);
else 
  save_root_folder = fullfile('/rmt/work/deeplabel/exper/voc12/res', feature_name, model_name, testset, 'crf', post_folder);
end

fprintf(1, 'Saving to %s\n', save_root_folder);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to chage values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
seg_res_dir = [save_root_folder '/results/VOC2012/'];
save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset '_cls']);

seg_root = fullfile(VOC_root_folder, 'VOC2012');

if ~exist(save_result_folder, 'dir')
    mkdir(save_result_folder);
end

VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset);

if is_mat
  % crop the results
  load('pascal_seg_colormap.mat');

  output_dir = dir(fullfile(output_mat_folder, '*.mat'));

  for i = 1 : numel(output_dir)
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
    [~, result] = max(result, [], 3);
    result = uint8(result) - 1;
    imwrite(result, colormap, fullfile(save_result_folder, [img_fn, '.png']));
  end
end

% get iou score
if strcmp(testset, 'val')
  [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);
else
  fprintf(1, 'This is test set. No evaluation. Just saved as png\n');
end 

    
    

