#include <fstream>
#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, softmax_top_vec_);

  SoftmaxLossParameter softmaxloss_param = this->layer_param_.softmaxloss_param();

  // read the weight for each class
  if (softmaxloss_param.has_weight_source()) {
    const string& weight_source = softmaxloss_param.weight_source();
    LOG(INFO) << "Opening file " << weight_source;
    std::fstream infile(weight_source.c_str(), std::fstream::in);
    CHECK(infile.is_open());

    Dtype tmp_val;
    while (infile >> tmp_val) {
      CHECK_GE(tmp_val, 0) << "Weights cannot be negative";
      loss_weights_.push_back(tmp_val);
    }
    infile.close();    

    CHECK_EQ(loss_weights_.size(), prob_.channels());
  } else {
    LOG(INFO) << "Weight_Loss file is not provided. Assign all one to it.";
    loss_weights_.assign(prob_.channels(), 1.0);
  }
  for (int c = 0; c < softmaxloss_param.ignore_label_size(); ++c){
    ignore_label_.insert(softmaxloss_param.ignore_label(c));
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  softmax_layer_->Reshape(softmax_bottom_vec_, softmax_top_vec_);
  if (top.size() >= 2) {
    // softmax output
    top[1]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  int channels = prob_.channels();
  Dtype batch_weight = 0;
  int spatial_dim = prob_.height() * prob_.width();
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      const int gt_label = static_cast<int>(label[i * spatial_dim + j]);
      //printf("gt_label: %d \n", gt_label);

      if (ignore_label_.count(gt_label) != 0) {
	// ignore the pixel with this gt_label
	continue;
      } else if (gt_label >= 0 && gt_label < channels) {
	batch_weight += loss_weights_[gt_label];
	// weighted loss
	loss -= loss_weights_[gt_label] * log(std::max(prob_data[i * dim +
           gt_label * spatial_dim + j], Dtype(FLT_MIN)));
      } else {
	LOG(FATAL) << "Unexpected label " << gt_label;
      }
    }
  }
  top[0]->mutable_cpu_data()[0] = loss / batch_weight;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    // Ignore this
    /*
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
    */
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    // We need to do *weighted* copy, so we defer that for the loop
    //caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    int channels = prob_.channels();
    Dtype batch_weight = 0;
    int spatial_dim = prob_.height() * prob_.width();
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
	const int gt_label = static_cast<int>(label[i * spatial_dim + j]);

	if (ignore_label_.count(gt_label) != 0) {
	  // ignore the pixel with this gt_label
	  continue;
	} else if (gt_label >= 0 && gt_label < channels) {
	  batch_weight += loss_weights_[gt_label];
	  for (int c = 0; c < channels; ++c) {
	    bottom_diff[i * dim + c * spatial_dim + j] = 
	      loss_weights_[gt_label] * prob_data[i * dim + c * spatial_dim + j];
	  }
	  bottom_diff[i * dim + gt_label * spatial_dim + j] -= 
	    loss_weights_[gt_label];
	} else {
	  LOG(ERROR) << "Unexpected label.";
	}
      }
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(prob_.count(), loss_weight / batch_weight, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SOFTMAX_LOSS, SoftmaxWithLossLayer);
}  // namespace caffe
