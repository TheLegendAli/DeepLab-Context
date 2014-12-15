clear all; close all;

img_folder = '../VOC2012/JPEGImages';

data_folder = '../VOC2012/ImageSets/Segmentation';
data_fn = {'VOC2012_test.txt', ...
          'VOC2012_trainval_aug.txt'};
      
for i = 1 : numel(data_fn)
    list = GetList(fullfile(data_folder, data_fn{i}));
    
    min_dim = 10000*ones(1, 2);
    max_dim = zeros(1, 2);
   
    for j = 1 : numel(list)
        fprintf(1, 'processing %d (%d) ...\n', j, numel(list));
        
        img = imread(fullfile(img_folder, [list{j} '.jpg']));

        img_row = size(img, 1);
        img_col   = size(img, 2);

        if img_row < min_dim(1)
            min_dim(1) = img_row;
        end

        if img_col < min_dim(2)
            min_dim(2) = img_col;
        end

        if img_row > max_dim(1)
            max_dim(1) = img_row;
        end

        if img_col > max_dim(2)
            max_dim(2) = img_col;
        end
    end

    max_dim
    min_dim
    
end

