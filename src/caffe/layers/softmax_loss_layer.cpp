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

  // read the weight for each class
  if (this->layer_param_.softmaxloss_param().has_weight_source()) {
    const string& weight_source = this->layer_param_.softmaxloss_param().weight_source();
    LOG(INFO) << "Opening file " << weight_source;
    std::fstream infile(weight_source.c_str(), std::fstream::in);
    if (!infile.is_open()){
      LOG(INFO) << "Fail to open " << weight_source << ". Assign all one to weight loss.";
      loss_weights_.assign(prob_.channels(), 1.0);
    }        

    Dtype tmp_val;
    while (infile >> tmp_val) {
      loss_weights_.push_back(tmp_val);
    }
    infile.close();    

    CHECK_EQ(loss_weights_.size(), prob_.channels());
  } else {
    LOG(INFO) << "Weight_Loss file is not provided. Assign all one to it.";
    loss_weights_.assign(prob_.channels(), 1.0);
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
  // jay add
  int channels = prob_.channels();
  int valid_pixel_count = 0;
  // end jay
  int spatial_dim = prob_.height() * prob_.width();
  Dtype loss = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      const int gt_label = static_cast<int>(label[i * spatial_dim + j]);

      if (gt_label < channels) {
	++valid_pixel_count;

	CHECK_GT(dim, gt_label * spatial_dim);

	// weighted loss
	loss -= loss_weights_[gt_label] * log(std::max(prob_data[i * dim +
           gt_label * spatial_dim + j], Dtype(FLT_MIN)));
      }
    }
  }

  // jay add
  top[0]->mutable_cpu_data()[0] = loss / valid_pixel_count;
  // end jay
  //top[0]->mutable_cpu_data()[0] = loss / num / spatial_dim;
  if (top.size() == 2) {
    top[1]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = bottom[1]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;

    // jay add
    int channels = prob_.channels();
    int valid_pixel_count = 0;
    // end jay

    int spatial_dim = prob_.height() * prob_.width();
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
	// jay add
	const int gt_label = static_cast<int>(label[i * spatial_dim + j]);
	if (gt_label < channels) {
	  bottom_diff[i * dim + gt_label * spatial_dim + j] -= 
                                             loss_weights_[gt_label];
	  ++valid_pixel_count;
	}
	// end jay

        //bottom_diff[i * dim + static_cast<int>(label[i * spatial_dim + j])
        //    * spatial_dim + j] -= 1;
      }
    }

    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    // jay add
    caffe_scal(prob_.count(), loss_weight / valid_pixel_count, bottom_diff);
    //caffe_scal(prob_.count(), loss_weight / num / spatial_dim, bottom_diff);
    // end jay
  }
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossLayer);
REGISTER_LAYER_CLASS(SOFTMAX_LOSS, SoftmaxWithLossLayer);
}  // namespace caffe
