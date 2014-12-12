#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/interp.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void InterpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  InterpParameter interp_param = this->layer_param_.interp_param();
  height_out_ = interp_param.height();
  width_out_  = interp_param.width();
  CHECK_GT(height_out_, 0) << "Need to specify height and this should be positive";
  CHECK_GT(width_out_, 0) << "Need to specify width and this should be positive";
}

template <typename Dtype>
void InterpLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  channels_ = bottom[0]->channels();
  height_in_ = bottom[0]->height();
  width_in_ = bottom[0]->width();
  top[0]->Reshape(num_, channels_, height_out_, width_out_);
}

template <typename Dtype>
void InterpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  caffe_cpu_interp2<Dtype,false>(num_ * channels_,
    bottom[0]->cpu_data(), 0, 0, height_in_, width_in_, height_in_, width_in_,
    top[0]->mutable_cpu_data(), 0, 0, height_out_, width_out_, height_out_, width_out_);
}

template <typename Dtype>
void InterpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  caffe_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_cpu_diff());
  caffe_cpu_interp2_backward<Dtype,false>(num_ * channels_,
    bottom[0]->mutable_cpu_diff(), 0, 0, height_in_, width_in_, height_in_, width_in_,
    top[0]->cpu_diff(), 0, 0, height_out_, width_out_, height_out_, width_out_);
}

template <typename Dtype>
void InterpLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  caffe_gpu_interp2<Dtype,false>(num_ * channels_,
    bottom[0]->gpu_data(), 0, 0, height_in_, width_in_, height_in_, width_in_,
    top[0]->mutable_gpu_data(), 0, 0, height_out_, width_out_, height_out_, width_out_);
}

template <typename Dtype>
void InterpLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) { return; }
  caffe_gpu_set(bottom[0]->count(), Dtype(0), bottom[0]->mutable_gpu_diff());
  caffe_gpu_interp2_backward<Dtype,false>(num_ * channels_,
    bottom[0]->mutable_gpu_diff(), 0, 0, height_in_, width_in_, height_in_, width_in_,
    top[0]->gpu_diff(), 0, 0, height_out_, width_out_, height_out_, width_out_);
}

INSTANTIATE_CLASS(InterpLayer);
REGISTER_LAYER_CLASS(INTERP, InterpLayer);

}  // namespace caffe
