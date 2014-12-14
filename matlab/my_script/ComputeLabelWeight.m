% compute the ground truth label weight, which is propotional to the label
% counts
%
clear all; close all;

num = 21;  % number of class
labelcounts = zeros(1, num);
count=0;

save_folder = '~/workspace/deeplabeling/examples/segnet';
save_fn = {'loss_weight_train.txt', ...
          'loss_weight_val.txt', ...
          'loss_weight_train_aug.txt', ...
          'loss_weight_trainval_aug.txt'};

data_folder = '../VOC2012/ImageSets/Segmentation';
data_fn = {'VOC2012_train.txt', ...
          'VOC2012_val.txt', ...
          'VOC2012_train_aug.txt', ...
          'VOC2012_trainval_aug.txt'};

img_folder = '../VOC2012/SegmentationClassAug';

for i = 1 : numel(data_fn)
    list = GetList(fullfile(data_folder, data_fn{i}));
    
    % accumulate the label counts
    for j = 1 : numel(list)
        fprintf(1, 'processing %d (%d) ...\n', j, numel(list));
        
        gtim = imread(fullfile(img_folder, [list{j} '.png']));
        gtim = double(gtim);
        
        %pixel locations to include in computation
        locs = gtim<255;

        % joint histogram
        sumim = 1+gtim;
        hs = histc(sumim(locs),1:num); 
        count = count + numel(find(locs));
        labelcounts(:) = labelcounts(:) + hs(:);
    end

    % compute the inverse of label counts
    freqs = labelcounts / count;
    c = num / (sum( 1./ freqs));
    weights = c ./ freqs;
    
    % save
    fid = fopen(fullfile(save_folder, save_fn{i}), 'w');
    fprintf(fid, '%1.4f\n', weights);    
    fclose(fid);
end