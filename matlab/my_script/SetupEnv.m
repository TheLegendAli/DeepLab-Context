% set up the environment variables
%

clear all; close all;
load('./pascal_seg_colormap.mat');

is_server       = 1;

crf_load_mat    = 1;   % the densecrf code load MAT files directly (no call SaveMatAsBin.m)
                       % used ONLY by DownSampleFeature.m
learn_crf       = 0;   % is the crf parameters learned or cross-validated

is_mat          = 1;   % the results to be evaluated are saved as mat or png
has_postprocess = 0;   % has done densecrf post processing or not
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

% vgg128_noup_pool3_strongweak6
% bi_w = 35, bi_x_std = 85, bi_r_std = 5, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_strongweak200
% bi_w = 10, bi_x_std = 80, bi_r_std = 5, pos_w = 3, pos_x_std = 10

% vgg128_noup_pool3_strongweak500
% bi_w = 9, bi_x_std = 77, bi_r_std = 5, pos_w = 5, pos_x_std = 5

% vgg128_noup_pool3_strongweak750
% bi_w = 9, bi_x_std = 59, bi_r_std = 5, pos_w = 5, pos_x_std = 5

% vgg128_noup_pool3_strongweak1000
% bi_w = 7, bi_x_std = 75, bi_r_std = 5, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_strongweak5K
% bi_w = 5, bi_x_std = 100, bi_r_std = 5, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_small
% bi_w = 5, bi_x_std = 61, bi_r_std = 3, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_cocomap2
% bi_w = 5, bi_x_std = 57, bi_r_std = 5, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_strongweak_cocomapbg
% bi_w = 5, bi_x_std = 63, bi_r_std = 5, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_strongstrong5K_cocomapbg
% bi_w = 5, bi_x_std = 63, bi_r_std = 5, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_strongstrongweak_cocomapbg3
% bi_w = 5, bi_x_std = 73, bi_r_std = 5, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_strongstrongweak_cocomapbg4
% bi_w = 5, bi_x_std = 71, bi_r_std = 3, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_20M (not xvaled)
% bi_w = 5, bi_x_std = 71, bi_r_std = 3, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_125M 
% bi_w = 4, bi_x_std = 72, bi_r_std = 5, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_largewin
% bi_w = 4, bi_x_std = 56, bi_r_std = 4, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_largewin2
% bi_w = 4, bi_x_std = 53, bi_r_std = 5, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_20M_largewin
% bi_w = 4, bi_x_std = 123, bi_r_std = 5, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_20M_largewin2
% bi_w = 4, bi_x_std = 175, bi_r_std = 5, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_20M_largewin3
% bi_w = 4, bi_x_std = 121, bi_r_std = 5, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_largewin_coco
% bi_w = 4, bi_x_std = 65, bi_r_std = 5, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_40M_largewin_spm
% bi_w = 4, bi_x_std = 83, bi_r_std = 4, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_17M_largewin2
% bi_w = 4, bi_x_std = 87, bi_r_std = 5, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_20M_largewin2_coco
% bi_w = 4, bi_x_std = 65, bi_r_std = 5, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_20M_largewin3_coco
% bi_w = 5, bi_x_std = 69, bi_r_std = 5, pos_w = 3, pos_x_std = 3

% vgg128_noup_pool3_125M_coco
% bi_w = 5, bi_x_std = 49, bi_r_std = 3, pos_w = 3, pos_x_std = 3



%
% alexnet_noup_pool3_7M
% bi_w = 5, bi_x_std = 61, bi_r_std = 5, pos_w = 3, pos_x_std = 3


% erode_gt (bbox)
% bi_w = 41, bi_x_std = 33, bi_r_std = 4

% erode_gt_mask
% bi_w = 39, bi_x_std = 23, bi_r_std = 4

% erode_gt/bboxErode20
% bi_w = 45, bi_x_std = 37, bi_r_std = 3, pos_w = 15, pos_x_std = 3
 

%
% initial or default values for crf
bi_w           = 5; 
bi_x_std       = 49;
bi_r_std       = 3;

pos_w          = 3;
pos_x_std      = 3;

% used for learn_crf
model_type     = 1;  % 0: Potts, 1: Diagonal, 2: Matrix label compacitability
epoch          = 1;  % used for learned crf parameters
%

%
dataset    = 'voc12';  %'voc12', 'coco'
trainset   = 'train_aug';       % not used
testset    = 'test';            %'val', 'test'

model_name = 'vgg128_noup_pool3_20M_largewin3_coco_cls_baseline'; 

feature_name = 'features2';
feature_type = 'fc8'; % fc8 / crf

% method to get "scores" for image classification task
cls_score_type = 'hard';    % 'hard', 'soft', 'score'

% feature_name = 'erode_gt';     % 'erode_gt', 'features', 'features4', 'features2', ''
% feature_type = 'bboxErode20_OccluBias';        %'bboxErode20', 'fc8', 'crf', 'fc8_crf'


id                 = 'comp6';
seg_id             = id;
seg_task_folder    = 'Segmentation';
seg_gt_task_folder = 'SegmentationClass';

cls_id             = 'comp2';
cls_task_folder    = 'Main';
cls_gt_task_folder = 'Main';

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
range_pos_x_std = [3];

range_bi_w = [5];
range_bi_x_std = [49];
range_bi_r_std = [4 5];


