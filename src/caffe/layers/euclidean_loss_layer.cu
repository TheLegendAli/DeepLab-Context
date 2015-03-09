#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  caffe_gpu_scal(count, Dtype(1 / sqrt(count)), diff_.mutable_gpu_data());
  Dtype dot;
  caffe_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
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
void EuclideanLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
      caffe_gpu_axpby(
          count,                              // count
          alpha,                              // alpha
          diff_.gpu_data(),                   // a
          Dtype(0),                           // beta
          bottom[i]->mutable_gpu_diff());  // b
    }
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(EuclideanLossLayer);

}  // namespace caffe
