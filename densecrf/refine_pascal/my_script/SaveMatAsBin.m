% save mat score maps as bin file for cpp
% 

mat_folder = '../feature';
img_folder = '../img';

save_folder = '../feature_bin';

if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

mat_dir = dir(fullfile(mat_folder, '*.mat'));

for i = 1 : numel(mat_dir)
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