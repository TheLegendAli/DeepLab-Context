% compute the ground truth label weight, which is propotional to the label
% counts
%
clear all; close all;

dataset = 'VOC2012';  %VOC2012, coco

if strcmp(dataset, 'VOC2012')
    num = 21;  % number of class
    save_folder = '~/dataset/PASCAL/VOCdevkit/VOC2012/list';
    save_fn = {'loss_weight_train.txt', ...
          'loss_weight_val.txt', ...
          'loss_weight_train_aug.txt', ...
          'loss_weight_trainval_aug.txt'};
      
    data_folder = '~/dataset/PASCAL/VOCdevkit/VOC2012/ImageSets/Segmentation';
    data_fn = {'VOC2012_train.txt', ...
          'VOC2012_val.txt', ...
          'VOC2012_train_aug.txt', ...
          'VOC2012_trainval_aug.txt'};

    img_folder = '~/dataset/PASCAL/VOCdevkit/VOC2012/SegmentationClassAug';
else
    num = 91;
    save_folder = '~/dataset/coco/list';
    save_fn = {'loss_weight_train2014.txt', ...
        'loss_weight_val2014.txt', ...
        'loss_weight_trainval2014.txt'};
    data_folder = '~/dataset/coco/ImageSets/Segmentation';
    data_fn = {'train2014.txt', ...
        'val2014.txt',...
        'trainval2014.txt'};
    img_folder = '~/dataset/coco/SegmentationClass';
end


if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

pairwise_cooccur = cell(1, numel(data_fn));

for i = 1 : numel(data_fn)
    list = GetList(fullfile(data_folder, data_fn{i}));

    labelcounts = zeros(1, num);
    count=0;
    
    pairwise_labelcounts = zeros(num, num);
    
    % accumulate the label counts
    for j = 1 : numel(list)
        if mod(j, 10000) == 0
            fprintf(1, 'processing %d (%d) ...\n', j, numel(list));
        end
        
        gtim = imread(fullfile(img_folder, [list{j} '.png']));
        gtim = double(gtim);
        
        %pixel locations to include in computation
        locs = gtim<255;

        % joint histogram
        sumim = 1+gtim;
        hs = histc(sumim(locs),1:num); 
        count = count + numel(find(locs));
        labelcounts(:) = labelcounts(:) + hs(:);
        
        % assume 1st element is background
        for m = 1 : num
            if hs(m) == 0
                continue;
            end
            for n = m+1 : num
                if hs(n) == 0
                    continue;
                end
                pairwise_labelcounts(m, n) = pairwise_labelcounts(m, n) + 1;                
                pairwise_labelcounts(n, m) = pairwise_labelcounts(n, m) + 1;                
                
            end
        end        
    end

    % compute the inverse of label counts
    labelcounts = labelcounts(labelcounts ~= 0);
    freqs = labelcounts / count;    
    c = num / (sum( 1./ freqs));
    weights = c ./ freqs;
      
    % save
    fid = fopen(fullfile(save_folder, save_fn{i}), 'w');
    fprintf(fid, '%1.12f\n', weights);    
    fclose(fid);
    
    fn = strrep(save_fn{i}, 'weight', 'freq');
    fid = fopen(fullfile(save_folder, fn), 'w');
    fprintf(fid, '%1.12f\n', freqs);
    fclose(fid);
    
    % self compactable
    pairwise_cooccur{i} = ones(size(pairwise_labelcounts));
    
    for kk = 1 : size(pairwise_labelcounts, 1)
        pairwise_cooccur{i}(kk, kk) = 0;
        pairwise_labelcounts(kk, kk) = 1;
    end
    
    % inf penalty for no cooccurrence
    ind = pairwise_labelcounts == 0;
    pairwise_cooccur{i}(ind) = 1e10;
    
    fn = strrep(save_fn{i}, 'loss', 'pairwise');
    fid = fopen(fullfile(save_folder, fn), 'w');
    fprintf(fid, '%d\n', pairwise_cooccur{i});
    fclose(fid);
    
end