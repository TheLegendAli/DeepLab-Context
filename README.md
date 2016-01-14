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

    @article{papandreou15weak,
      title={Weakly- and Semi-Supervised Learning of a DCNN for Semantic Image Segmentation},
      author={George Papandreou and Liang-Chieh Chen and Kevin Murphy and Alan L Yuille},
      journal={arxiv:1502.02734},
      year={2015}
    }

Note that if you use the densecrf implementation, please consult and cite the following paper:

    @inproceedings{KrahenbuhlK11,
      title={Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials},
      author={Philipp Kr{\"{a}}henb{\"{u}}hl and Vladlen Koltun},
      booktitle={NIPS},      
      year={2011}
    }

### Performance

DeepLab currently achieves 73.9% on the challenging PASCAL VOC 2012 image segmentation task -- see the [leaderboard](http://host.robots.ox.ac.uk:8080/leaderboard/displaylb.php?challengeid=11&compid=6). 

### Pre-trained models

We have released several trained models and corresponding prototxt files at [here](http://ccvl.stat.ucla.edu/software/deeplab/). Please check it for more model details.

The best model among the released ones yields 73.6% on PASCAL VOC 2012 test set.

### Python wrapper requirements

1. Install wget library for python
```
sudo pip install wget
```
2. Change DATA_ROOT to point to the PASCAL images

3. To use the mat_read_layer and mat_write_layer, please download and install [matio](http://sourceforge.net/projects/matio/files/matio/1.5.2/).

### Running the code

```
python run.py
```

### FAQ

Check [FAQ](http://ccvl.stat.ucla.edu/deeplab_faq/) if you have some problems while using the code.
