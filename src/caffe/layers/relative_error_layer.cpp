#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void RelativeErrorLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
}

template <typename Dtype>
void RelativeErrorLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  top[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void RelativeErrorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Blob<Dtype> diff;
  Dtype nom, denom;
  diff.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  int count = diff.count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff.mutable_cpu_data());
  const Dtype alpha = Dtype(1 / sqrt(count));
  caffe_scal(count, alpha, diff.mutable_cpu_data());
  nom = caffe_cpu_dot(count, diff.cpu_data(), diff.cpu_data());
  caffe_cpu_scale(count, alpha, bottom[1]->cpu_data(), diff.mutable_cpu_data());
  denom = caffe_cpu_dot(count, diff.cpu_data(), diff.cpu_data());
  denom = std::max(denom, Dtype(1e-6));
  top[0]->mutable_cpu_data()[0] = sqrt(nom / denom);
}

#ifndef CPU_ONLY
template <typename Dtype>
void RelativeErrorLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Blob<Dtype> diff;
  Dtype nom, denom;
  diff.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  int count = diff.count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff.mutable_gpu_data());
  const Dtype alpha = Dtype(1 / sqrt(count));
  caffe_gpu_scal(count, alpha, diff.mutable_gpu_data());
  caffe_gpu_dot(count, diff.gpu_data(), diff.gpu_data(), &nom);
  caffe_gpu_scale(count, alpha, bottom[1]->gpu_data(), diff.mutable_gpu_data());
  caffe_gpu_dot(count, diff.gpu_data(), diff.gpu_data(), &denom);
  denom = std::max(denom, Dtype(1e-6));
  top[0]->mutable_cpu_data()[0] = sqrt(nom / denom);
}
#endif

#ifdef CPU_ONLY
STUB_GPU_FORWARD(RelativeErrorLayer,Forward);
#endif

INSTANTIATE_CLASS(RelativeErrorLayer);
REGISTER_LAYER_CLASS(RELATIVE_ERROR, RelativeErrorLayer);
}  // namespace caffe
