% downsample the bin files for faster cross-validation and not overfit val set
% 
clear all; 

down_sample_rate = 8;

is_server = 1;

dataset = 'voc12';   %voc12, coco

testset = 'val';
model_name = 'vgg128_noup_pool3_coco';  %vgg128_noup, vgg128_ms
feature_name = 'features';
feature_type = 'fc8';

if is_server
  mat_folder  = fullfile('/rmt/work/deeplabel/exper', dataset, feature_name, model_name, testset, feature_type);
  save_folder = fullfile(mat_folder, 'bin');
else
  mat_folder  = '../feature';
  save_folder = '../feature_bin';
end

dest_folder = [save_folder, sprintf('_downsampleBy%d', down_sample_rate)];

if ~exist(dest_folder, 'dir')
  mkdir(dest_folder)
end

save_dir = dir(fullfile(save_folder, '*.bin'));

save_dir = save_dir(1:down_sample_rate:end);

for i = 1 : numel(save_dir)
  fprintf(1, 'processing %d (%d)...\n', i, numel(save_dir));
  copyfile(fullfile(save_folder, save_dir(i).name), fullfile(dest_folder, save_dir(i).name));
end
