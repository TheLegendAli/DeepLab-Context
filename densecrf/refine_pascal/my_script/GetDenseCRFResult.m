clear all;

load('pascal_seg_colormap');

is_server = 1;

id = 'comp6';
trainset = 'train_aug';
testset  = 'val';

model_name = 'vgg128_ms'; %vgg128_noup

if is_server
  map_folder = fullfile('/rmt/work/deeplabel/exper/voc12/res', model_name, testset, 'fc8', 'post_densecrf');
else 
  map_folder = '../result';
end

map_dir = dir(fullfile(map_folder, '*.bin'));

save_root_folder = map_folder;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to change values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
seg_res_dir = [save_root_folder '/results/VOC2012/'];
save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset '_cls']);

if ~exist(save_result_folder, 'dir')
    mkdir(save_result_folder);
end

for i = 1 : numel(map_dir)
    fprintf(1, 'processing %d (%d)...\n', i, numel(map_dir));
    map = LoadBinFile(fullfile(map_folder, map_dir(i).name), 'int16');

    img_fn = map_dir(i).name(1:end-4);
    imwrite(uint8(map), colormap, fullfile(save_result_folder, [img_fn, '.png']));
end
