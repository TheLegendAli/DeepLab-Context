#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  caffe_scal(count, Dtype(1 / sqrt(count)), diff_.mutable_cpu_data());
  Dtype dot = caffe_cpu_dot(count, diff_.cpu_data(), diff_.cpu_data());
  switch (this->layer_param_.euclidean_loss_param().type()) {
  case EuclideanLossParameter_Type_L2:
    loss_ = 0.5 * dot;
    break;
  case EuclideanLossParameter_Type_L2sqrt:
    loss_ = sqrt(dot);
    break;
  default:
    LOG(FATAL) << "Unknown Type";
  }
  top[0]->mutable_cpu_data()[0] = loss_;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  int count = bottom[0]->count();
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      Dtype alpha;
      switch (this->layer_param_.euclidean_loss_param().type()) {
      case EuclideanLossParameter_Type_L2:
	alpha = sign * top[0]->cpu_diff()[0] / sqrt(count);
	break;
      case EuclideanLossParameter_Type_L2sqrt:
	alpha = sign * top[0]->cpu_diff()[0] / sqrt(count) /
	  std::max(loss_, Dtype(1e-6));
	break;
      default:
	LOG(FATAL) << "Unknown Type";
      }
      caffe_cpu_axpby(
          count,                              // count
          alpha,                              // alpha
          diff_.cpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_cpu_diff());  // b
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossLayer);
REGISTER_LAYER_CLASS(EUCLIDEAN_LOSS, EuclideanLossLayer);
}  // namespace caffe
