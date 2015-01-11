#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void BiasChannelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  BiasChannelParameter param = this->layer_param_.bias_channel_param();
  bg_bias_ = param.bg_bias();
  fg_bias_ = param.fg_bias();
  CHECK_GT(bg_bias_, 0) << "BG bias needs to be positive";
  CHECK_GT(fg_bias_, bg_bias_) << "FG bias needs to be larger than BG bias";
  for (int i = 0; i < param.ignore_label_size(); ++i){
    ignore_label_.insert(param.ignore_label(i));
  }
}

template <typename Dtype>
void BiasChannelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  //
  CHECK_EQ(bottom[1]->num(), num_) << "Input channels incompatible in num";
  max_labels_ = bottom[1]->channels();
  CHECK_GE(max_labels_, 1) << "Label blob needs to be non-empty";
  CHECK_EQ(bottom[1]->height(), 1) << "Label height";
  CHECK_EQ(bottom[1]->width(), 1) << "Label width";
  //
  top[0]->Reshape(num_, channels_, height_, width_);
}

template <typename Dtype>
void BiasChannelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  caffe_copy(bottom[0]->count(), bottom[0]->cpu_data(), top[0]->mutable_cpu_data());
  for (int n = 0; n < num_; ++n) {
    // Bias the background score
    caffe_add_scalar(height_ * width_, bg_bias_, top[0]->mutable_cpu_data(n, 0));
    // Bias the foreground scores
    for (int j = 0; j < max_labels_; ++j) {
      const int label = static_cast<int>(*bottom[1]->cpu_data(n, j));
      if (ignore_label_.count(label) != 0) {
	continue;
      } else if (label > 0 && label < channels_) {
	caffe_add_scalar(height_ * width_, fg_bias_, top[0]->mutable_cpu_data(n, label));	
      } else {
	LOG(FATAL) << "Unexpected label " << label;
      }
    }
  }
}

template <typename Dtype>
void BiasChannelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << "Cannot propagate down to label input";
  }
  if (propagate_down[0]) {
    caffe_copy(bottom[0]->count(), top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff());
  }
}

template <typename Dtype>
void BiasChannelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  caffe_copy(bottom[0]->count(), bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
  for (int n = 0; n < num_; ++n) {
    // Bias the background score
    caffe_gpu_add_scalar(height_ * width_, bg_bias_, top[0]->mutable_gpu_data(n, 0));
    // Bias the foreground scores
    for (int j = 0; j < max_labels_; ++j) {
      const int label = static_cast<int>(*bottom[1]->cpu_data(n, j));
      if (ignore_label_.count(label) != 0) {
	continue;
      } else if (label > 0 && label < channels_) {
	caffe_gpu_add_scalar(height_ * width_, fg_bias_, top[0]->mutable_gpu_data(n, label));	
      } else {
	LOG(FATAL) << "Unexpected label " << label;
      }
    }
  }
}

template <typename Dtype>
void BiasChannelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << "Cannot propagate down to label input";
  }
  if (propagate_down[0]) {
    caffe_copy(bottom[0]->count(), top[0]->gpu_diff(), bottom[0]->mutable_gpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(BiasChannelLayer);
#endif

INSTANTIATE_CLASS(BiasChannelLayer);
}  // namespace caffe
