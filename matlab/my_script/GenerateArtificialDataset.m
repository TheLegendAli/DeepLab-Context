gt_folder = '../SegmentationClassAug_Tmp';
save_folder = '../SegmentationClassAug_DownSample';

if ~exist(save_folder, 'dir')
    mkdir(save_folder)
end

gts = dir(fullfile(gt_folder, '*.png'));

for i = 1 : numel(gts)
    gt = imread(fullfile(gt_folder, gts(i).name));
    
    fprintf(1, 'pre-dim %d %d\n', size(gt,1), size(gt,2));
    
    gt = ExtractRegion(gt, 5, 1);
    gt = ExtractRegion(gt, 2, 2);            
    gt = ExtractRegion(gt, 5, 1);
    gt = ExtractRegion(gt, 2, 2);
    
    fprintf(1, 'after-dim %d %d\n', size(gt,1), size(gt,2));
    
    imwrite(gt, fullfile(save_folder, gts(i).name));
end