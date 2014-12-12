addpath('/rmt/work/deeplabeling/matlab/caffe');

% vgg-16 net
model_def_file = '/rmt/work/deeplabeling/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers_deploy.prototxt';
model_file = '/rmt/work/deeplabeling/models/VGG_ILSVRC_16_layers/VGG_ILSVRC_16_layers.caffemodel';
use_gpu = true;
im = imread('/rmt/work/deeplabeling/examples/images/cat.jpg');

vgg_scores = matcaffe_demo_vgg_mean_pix(im, use_gpu, model_def_file, model_file);
vgg_scores = mean(vgg_scores,2);
[~,maxlabel] = max(vgg_scores);

