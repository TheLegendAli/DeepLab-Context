clear all;

load('pascal_seg_colormap');

is_server = 1;
down_sample_rate = 8;

pos_w          = 3;
pos_x_std      = 3;

bi_w           = 5;
bi_x_std       = 50;
bi_r_std       = 10;

id = 'comp6';
trainset = 'train_aug';
testset  = 'val';

model_name = 'vgg128_ms_pool3'; %vgg128_noup, vgg128_ms

for w = 3:2:5         %[3 5 7 9 11]                %0.5:0.5:6  %[1 5 10 15 20]
  bi_w = w;
  for x_std = 80:5:100   %35:5:65               % [10 20 30 40 50]
    bi_x_std = x_std;
    for r_std = [3 5 7]   %5:5:10    % 5:5:20              % [10 20 30 40 50]
      bi_r_std = r_std;
    
      crf_folder = sprintf('post_densecrf_W%d_XStd%d_RStd%d_PosW%d_PosXStd%d_downsampleBy%d', bi_w, bi_x_std, bi_r_std, pos_w, pos_x_std, down_sample_rate);

      %if w==1 || w==2 || w == 3 || w==4 || w==5 || w==6
      %  crf_folder = sprintf('post_densecrf_PosW%d_PosXStd%d_downsampleBy%d', w, x_std, down_sample_rate); 
      %else
      %  crf_folder = sprintf('post_densecrf_PosW%1.1f_PosXStd%d_downsampleBy%d', w, x_std, down_sample_rate);
      %end

      if is_server
        map_folder = fullfile('/rmt/work/deeplabel/exper/voc12/res', model_name, testset, 'fc8', crf_folder);
      else 
        map_folder = '../result';
      end

      map_dir = dir(fullfile(map_folder, '*.bin'));

      save_root_folder = map_folder;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% You do not need to change values below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      seg_res_dir = [save_root_folder '/results/VOC2012/'];
      save_result_folder = fullfile(seg_res_dir, 'Segmentation', [id '_' testset '_cls']);

      if ~exist(save_result_folder, 'dir')
        mkdir(save_result_folder);
      end

      for i = 1 : numel(map_dir)
        fprintf(1, 'processing %d (%d)...\n', i, numel(map_dir));
        map = LoadBinFile(fullfile(map_folder, map_dir(i).name), 'int16');

        img_fn = map_dir(i).name(1:end-4);
        imwrite(uint8(map), colormap, fullfile(save_result_folder, [img_fn, '.png']));
      end
    end
  end
end
