## DeepLab

### Introduction

DeepLab is a state-of-art deep learning system for semantic image segmentation built on top of [Caffe](http://caffe.berkeleyvision.org).

It combines densely-computed deep convolutional neural network (CNN) responses with densely connected conditional random fields (CRF).

This distribution provides a publicly available implementation for the key model ingredients first reported in an [arXiv paper](http://arxiv.org/abs/1412.7062), accepted in revised form as conference publication to the ICLR-2015 conference. 
It also contains implementations for methods supporting model learning using only weakly labeled examples, described in a second follow-up [arXiv paper](http://arxiv.org/abs/1502.02734).
Please consult and consider citing the following papers:

    @inproceedings{chen14semantic,
      title={Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs},
      author={Liang-Chieh Chen and George Papandreou and Iasonas Kokkinos and Kevin Murphy and Alan L Yuille},
      booktitle={ICLR},
      url={http://arxiv.org/abs/1412.7062},
      year={2015}
    }

    @inproceedings{papandreou15weak,
      title={Weakly- and Semi-Supervised Learning of a DCNN for Semantic Image Segmentation},
      author={George Papandreou and Liang-Chieh Chen and Kevin Murphy and Alan L Yuille},
      url={http://arxiv.org/abs/1502.02734},
      year={2015}
    }

### Performance

At the time of its release, DeepLab is the state-of-art method on semantic image segmentation on the challenging PASCAL VOC-2012 image segmentation task, with the latest variant achieving 72.7% mean IoU on the test set -- see the [leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6).

### Pre-trained models

1. DeepLab and corresponding prototxt files at [here](http://www.cs.ucla.edu/~lcchen/deeplab-public/vgg128_noup/)

2. DeepLab-MSc at [here](http://www.cs.ucla.edu/~lcchen/deeplab-public/vgg128_ms_pool3/)

3. DeepLab-COCO (has fine-tuned on [MS-COCO](http://mscoco.org/) and then on PASCAL VOC [2012](http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2012/)) at [here](http://www.cs.ucla.edu/~lcchen/deeplab-public/vgg128_noup_pool3_cocomix/)

### Others

1. The scripts we used for our experiments:
    1. [run_pascal.sh](http://www.cs.ucla.edu/~lcchen/deeplab-public/run_pascal.sh): the script for training/testing on the PASCAL VOC 2012 dataset. __Note__ You also need to download this [file](http://www.cs.ucla.edu/~lcchen/deeplab-public/sub.sed)
    2. [run_densecrf.sh](http://www.cs.ucla.edu/~lcchen/deeplab-public/run_densecrf.sh) and [run_densecrf_grid_search.sh](http://www.cs.ucla.edu/~lcchen/deeplab-public/run_densecrf_grid_search.sh): the scripts we used for post-processing the DCNN computed results by DenseCRF.
2. The image list files used in our experiments:
    * The [list folder](http://www.cs.ucla.edu/~lcchen/deeplab-public/list) stores the list files for the PASCAL VOC 2012 dataset. You can download the zipped file [here](http://www.cs.ucla.edu/~lcchen/deeplab-public/list.zip) (i.e., all the lists).
    