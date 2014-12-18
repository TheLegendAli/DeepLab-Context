% save mat score maps as bin file for cpp
% 
clear all; 

is_server = 1;

testset = 'val';
model_name = 'vgg128_noup';   %vgg128_ms, vgg128_noup, vgg128_noup_glob
feature_name = 'features4';      %'features', 'features4'

if is_server
  mat_folder  = fullfile('/rmt/work/deeplabel/exper/voc12', feature_name, model_name, testset, 'fc8');
  img_folder  = '/rmt/data/pascal/VOCdevkit/VOC2012/JPEGImages';
  save_folder = fullfile(mat_folder, 'bin');
else
  mat_folder  = fullfile('~/workspace/deeplabeling/exper/voc12/features4', model_name, testset, 'fc8');
  img_folder  = '~/dataset/PASCAL/VOCdevkit/VOC2012/JPEGImages';
  save_folder = fullfile(mat_folder, 'bin');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to chage values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

mat_dir = dir(fullfile(mat_folder, '*.mat'));

for i = 1 : numel(mat_dir)
    fprintf(1, 'processing %d (%d)...\n', i, numel(mat_dir));
    data = load(fullfile(mat_folder, mat_dir(i).name));
    data = data.data;
    data = permute(data, [2 1 3]);    
    %Transform data to probability
    data = exp(data);
    data = bsxfun(@rdivide, data, sum(data, 3));
    
    img_fn = mat_dir(i).name(1:end-4);
    img_fn = strrep(img_fn, '_blob_0', '');
    img = imread(fullfile(img_folder, [img_fn, '.jpg']));
    img_row = size(img, 1);
    img_col = size(img, 2);
    
    data = data(1:img_row, 1:img_col, :);
    
    save_fn = fullfile(save_folder, [img_fn, '.bin']);
    
    SaveBinFile(data, save_fn, 'float');
end

