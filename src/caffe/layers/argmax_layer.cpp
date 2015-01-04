#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void ArgMaxLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  out_max_val_ = this->layer_param_.argmax_param().out_max_val();
  top_k_ = this->layer_param_.argmax_param().top_k();
  CHECK_GE(top_k_, 1) << " top k must not be less than 1.";
}

template <typename Dtype>
void ArgMaxLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  CHECK_LE(top_k_, channels_)
      << "top_k must be less than or equal to the number of channels.";
  // Produces max_ind and max_val if out_max_val_, otherwise only max_ind
  const int channels_out = (out_max_val_) ? 2 * top_k_ : top_k_;
  top[0]->Reshape(num_, channels_out, height_, width_);
}

template <typename Dtype>
void ArgMaxLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  for (int n = 0; n < num_; ++n) {
    for (int h = 0; h < height_; ++h) {
      for (int w = 0; w < width_; ++w) {
	const Dtype* bottom_data = bottom[0]->cpu_data(n, 0, h, w);
	const int channel_offset = height_ * width_;
	std::vector<std::pair<Dtype, int> > bottom_data_vector;
	for (int c = 0; c < channels_; ++c) {
	  bottom_data_vector.push_back(
	     std::make_pair(bottom_data[c * channel_offset], c));
	}
	std::partial_sort(
	    bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
	    bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());
	Dtype* top_data = top[0]->mutable_cpu_data(n, 0, h, w);
	for (int j = 0; j < top_k_; ++j) {
	  top_data[j * channel_offset] = bottom_data_vector[j].second;
	  if (out_max_val_) {
	    top_data[(top_k_ + j) * channel_offset] = bottom_data_vector[j].first;
	  }
	}
      }
    }
  }
}

INSTANTIATE_CLASS(ArgMaxLayer);
REGISTER_LAYER_CLASS(ARGMAX, ArgMaxLayer);

}  // namespace caffe
