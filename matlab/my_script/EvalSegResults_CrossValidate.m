clear all; close all;

% change values here
is_server        = 1;
down_sample_rate = 8;

pos_w          = 3;
pos_x_std      = 3;

bi_w           = 5;
bi_x_std       = 50;
bi_r_std       = 10;

id         = 'comp6';
%trainset  = 'trainval_aug';
trainset   = 'train_aug';

%testset   = 'trainval_aug';
testset    = 'val';

model_name = 'vgg128_ms_pool3';   %'vgg128_noup' or 'vgg128_ms'


if is_server
    VOC_root_folder = '/rmt/data/pascal/VOCdevkit';
else
    VOC_root_folder = '~/dataset/PASCAL/VOCdevkit';
end


best_avacc = -1;
best_w = -1;
best_x_std = -1;
best_r_std = -1;

fid = fopen(sprintf('cross_avgIOU_%s_%sDownSample%d.txt', model_name, testset, down_sample_rate), 'a');

for w = 3:2:5       %0.5:0.5:6 %[1 5 10 15 20]
  bi_w = w;
  for x_std = 80:5:100   %1:12 %[10 20 30 40 50]
    bi_x_std = x_std;
    for r_std = [3 5 7]  %5:5:10      %[10 20 30 40 50]
      bi_r_std = r_std;

      post_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d_downsampleBy%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std, down_sample_rate);

      %if w==1 || w==2 || w == 3 || w==4 || w==5 || w==6
      %  post_folder = sprintf('post_densecrf_PosW%d_PosXStd%d_downsampleBy%d', w, x_std, down_sample_rate); 
      %else
      %  post_folder = sprintf('post_densecrf_PosW%1.1f_PosXStd%d_downsampleBy%d', w, x_std, down_sample_rate);
      %end

      save_root_folder = fullfile('/rmt/work/deeplabel/exper/voc12/res', model_name, testset, 'fc8', post_folder);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to chage values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      seg_res_dir = [save_root_folder '/results/VOC2012/'];
      save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset '_cls']);

      seg_root = fullfile(VOC_root_folder, 'VOC2012');

      if ~exist(save_result_folder, 'dir')
        mkdir(save_result_folder);
      end

      VOCopts = GetVOCopts(seg_root, seg_res_dir, trainset, testset);

      % get iou score
      [accuracies, avacc, conf, rawcounts] = MyVOCevalseg(VOCopts, id);

      if best_avacc < avacc
        best_avacc = avacc;
        best_accuracies = accuracies;
        best_conf = conf;
        best_rawcounts = rawcounts;

        best_w = w;
        best_x_std = x_std;
        best_r_std = r_std;
        best_pos_w = pos_w;
        best_pos_x_std = pos_x_std;
      end

      fprintf(fid, 'w %2.2f, x_std %2.2f, r_std %2.2f, pos_w %2.2f, pos_x_std %2.2f, avacc %6.3f%%\n', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std, avacc);
      
    end
  end
end

    fprintf(1, 'Best avacc %6.3f%% occurs at w = %2.2f, x_std = %2.2f, r_std = %2.2f, pos_w %2.2f, pos_x_std %2.2f\n', best_avacc, best_w, best_x_std, best_r_std, best_pos_w, best_pos_x_std);

    

fclose(fid);
