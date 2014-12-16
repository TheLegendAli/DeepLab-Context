load('pascal_seg_colormap');

map_folder = '../result';

map_dir = dir(fullfile(map_folder, '*.bin'));

for i = 1 : numel(map_dir)
    map = LoadBinFile(fullfile(map_folder, map_dir(i).name), 'int16');
    
    figure(1), imshow(uint8(map), colormap);
end