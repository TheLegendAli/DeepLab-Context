% set up the environment variables
%

clear all; close all;
load('./pascal_seg_colormap.mat');

is_server       = 1;
is_mat          = 0;   % the results are saved as mat or png
has_postprocess = 1;   % has done densecrf post processing or not
is_argmax       = 0;   % the output has been taken argmax already (e.g., coco dataset). 
                       % assume the argmax takes C-convention (i.e., start from 0)

debug           = 0;   % if debug, show some results

% vgg128_noup_pool3 (not optimized well)
% bi_w = 5, bi_x_std = 50, bi_r_std = 10

% vgg128_ms_pool3
% bi_w = 3, bi_x_std = 95, bi_r_std = 3

% vgg128_noup_pool3_coco
% bi_w = 5, bi_x_std = 63, bi_r_std = 3

% vgg128_noup_pool3_cocomix
% bi_w = 5, bi_x_std = 67, bi_r_std = 3

% vgg128_noup_pool3_cocomix2
% bi_w = 4, bi_x_std = 75, bi_r_std = 3

% vgg128_ms_pool3_coco
% bi_w = 4, bi_x_std = 81, bi_r_std = 2

% vgg128_noup_pool3_bbox (no cross-validate)
% bi_w = 5, bi_x_std = 70, bi_r_std = 3

% vgg128_noup_pool3_bbox_crf
% bi_w = 7, bi_x_std = 41, bi_r_std = 7

% vgg128_noup_pool3_bbox_crf2
% bi_w = 5, bi_x_std = 77, bi_r_std = 3, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_adaweak (1st no cross-validate)
% bi_w = 5, bi_x_std = 50, bi_r_std = 10
% bi_w = 35, bi_x_std = 61, bi_r_std = 10

% vgg128_noup_pool3_strongweak2
% bi_w = 5, bi_x_std = 81, bi_r_std = 5, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_strongweak200
% bi_w = 10, bi_x_std = 80, bi_r_std = 5, pos_w = 3, pos_x_std = 10

% vgg128_noup_pool3_small
% bi_w = 5, bi_x_std = 61, bi_r_std = 3, pos_w = 3, pos_x_std = 3

% erode_gt (bbox)
% bi_w = 41, bi_x_std = 33, bi_r_std = 4

% erode_gt_mask
% bi_w = 39, bi_x_std = 23, bi_r_std = 4

% erode_gt/bboxErode20
% bi_w = 45, bi_x_std = 37, bi_r_std = 3, pos_w = 15, pos_x_std = 3
 
%
bi_w           = 10;   %5;   3
bi_x_std       = 80;   %67;   %50;  95
bi_r_std       = 5;    %3;    %10;  3

pos_w          = 3;     %3
pos_x_std      = 10;    %3

%
dataset    = 'voc12';  %'voc12', 'coco'
id         = 'comp6';
trainset   = 'train_aug';
testset    = 'val';            %'val', 'test'

model_name = 'vgg128_noup_pool3_strongweak200';   %'vgg128_noup', 'vgg128_noup_glob', 'vgg128_ms'

feature_name = 'features';
feature_type = 'fc8';

% feature_name = 'erode_gt';     % 'erode_gt', 'features', 'features4', 'features2', ''
% feature_type = 'bboxErode20_OccluBias';        %'bboxErode20', 'fc8', 'crf', 'fc8_crf'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% used for cross-validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
rng(10)

% downsampling files for cross-validation
down_sample_method = 2;      % 1: equally sample, 2: randomly pick num_sample
down_sample_rate   = 8;
num_sample         = 100;     % used for erode_gt 

% ranges for cross-validation
range_pos_w = [3];
range_pos_x_std = [10];

range_bi_w = 10;
range_bi_x_std = 80;
range_bi_r_std = [5];


