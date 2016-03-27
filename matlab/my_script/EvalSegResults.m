%SetupEnv;

function EvalSegResults(post_folder, feature_name, model_name, testset, feature_type, dataset, id, trainset, is_mat, is_argmax, has_postprocess)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to chage values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

VOC_root_folder = '{DATA_ROOT}';


output_mat_folder = fullfile('{ROOT}', dataset, feature_name, model_name, testset, feature_type);

save_root_folder = fullfile('{ROOT}', dataset, feature_name, model_name, testset, feature_type, post_folder);

fprintf(1, 'Saving to %s\n', save_root_folder);

if strcmp(dataset, '{EXP}')
  year_path = strcat('/results/', '{YEAR}', '/');
  seg_res_dir = [save_root_folder year_path];
  seg_root = VOC_root_folder;
  gt_dir   = fullfile(VOC_root_folder, 'SegmentationClass');
elseif strcmp(dataset, 'coco')
  seg_res_dir = [save_root_folder '/results/COCO2014/'];
  seg_root = fullfile(VOC_root_folder, '');
  gt_dir   = fullfile(VOC_root_folder, '', 'SegmentationClass');
end

save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset '_cls']);

if ~exist(save_result_folder, 'dir')
    mkdir(save_result_folder);
end

if strcmp(dataset, '{EXP}')
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, '{YEAR}');
elseif strcmp(dataset, 'coco')
  VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset, '');
end

if is_mat
  % crop the results
  load('pascal_seg_colormap.mat');

  output_dir = dir(fullfile(output_mat_folder, '*.mat'));

  for i = 1 : numel(output_dir)
    if mod(i, 100) == 0
        fprintf(1, 'processing %d (%d)...\n', i, numel(output_dir));
    end

    data = load(fullfile(output_mat_folder, output_dir(i).name));
    raw_result = data.data;
    raw_result = permute(raw_result, [2 1 3]);

    img_fn = output_dir(i).name(1:end-4);
    img_fn = strrep(img_fn, '_blob_0', '');
    
    if strcmp(dataset, '{EXP}')
      img = imread(fullfile(VOC_root_folder, 'JPEGImages', [img_fn, '.jpg']));
    elseif strcmp(dataset, 'coco')
      img = imread(fullfile(VOC_root_folder, 'JPEGImages', [img_fn, '.jpg']));
    end
    
    img_row = size(img, 1);
    img_col = size(img, 2);
    
    result = raw_result(1:img_row, 1:img_col, :);

    if ~is_argmax
      [~, result] = max(result, [], 3);
      result = uint8(result) - 1;
    else
      result = uint8(result);
    end

    if debug
        gt = imread(fullfile(gt_dir, [img_fn, '.png']));
        figure(1), 
        subplot(221),imshow(img), title('img');
        subplot(222),imshow(gt, colormap), title('gt');
        subplot(224), imshow(result,colormap), title('predict');
    end
    
    imwrite(result, colormap, fullfile(save_result_folder, [img_fn, '.png']));
  end
end

if has_postprocess == 2
    has_postprocess = 1;
end 

% get iou score
if strcmp(testset, 'val')
  [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id, has_postprocess);
else
  fprintf(1, 'This is test set. No evaluation. Just saved as png\n');
end 

    
    
