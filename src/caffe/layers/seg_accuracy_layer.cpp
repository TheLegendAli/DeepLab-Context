#include <algorithm>
#include <functional>
#include <utility>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SegAccuracyLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  top_k_ = this->layer_param_.accuracy_param().top_k();  
}

template <typename Dtype>
void SegAccuracyLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_LE(top_k_, bottom[0]->channels())
      << "top_k must be less than or equal to the number of channels (classes).";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
    << "The data and label should have the same number.";
  CHECK_EQ(bottom[1]->channels(), 1)
    << "The label should have one channel.";
  CHECK_EQ(bottom[0]->height(), bottom[1]->height())
    << "The data should have the same height as label.";
  CHECK_EQ(bottom[0]->width(), bottom[1]->width())
    << "The data should have the same width as label.";
  top[0]->Reshape(1, 1, 1, 1);
}

  /*
   * TODO: IOU accuracy
   *
   */
template <typename Dtype>
void SegAccuracyLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  Dtype accuracy = 0;
  const Dtype* bottom_data = bottom[0]->cpu_data();
  const Dtype* bottom_label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  //int dim = bottom[0]->count() / bottom[0]->num();
  int channels = bottom[0]->channels();
  int height = bottom[0]->height();
  int width = bottom[0]->width();
  vector<Dtype> maxval(top_k_+1);
  vector<int> max_id(top_k_+1);

  int data_index, label_index;
  int valid_pixel_count = 0;

  switch (this->layer_param_.seg_accuracy_param().metric()) {
  case SegAccuracyParameter_AccuracyMetric_PixelAccuracy:
    for (int i = 0; i < num; ++i) {
      for (int h = 0; h < height; ++h) {
	for (int w = 0; w < width; ++w) {
	  //Top-k accuracy
	  std::vector<std::pair<Dtype, int> > bottom_data_vector;
	  
	  for (int c = 0; c < channels; ++c) {
	    data_index = (c * height + h) * width + w;	    
	    bottom_data_vector.push_back(std::make_pair(bottom_data[data_index], c));
	  }
	  std::partial_sort(
	    bottom_data_vector.begin(), bottom_data_vector.begin() + top_k_,
	    bottom_data_vector.end(), std::greater<std::pair<Dtype, int> >());

	  // check if true label is in top k predictions

	  label_index = h * width + w;
	  const int gt_label = static_cast<int>(bottom_label[label_index]);
      
	  if (gt_label < channels) {
	    // current position is not "255", indicating ambiguous position
	    ++valid_pixel_count;

	    for (int k = 0; k < top_k_; k++) {
	      if (bottom_data_vector[k].second == gt_label) {
		++accuracy;
		break;
	      }
	    }
	  }
	}
      }

      bottom_data += bottom[0]->offset(1);
      bottom_label += bottom[1]->offset(1);
    }
    break;
  case SegAccuracyParameter_AccuracyMetric_ClassAccuracy:
    NOT_IMPLEMENTED;
    break;
  case SegAccuracyParameter_AccuracyMetric_PixelIOU:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown Segment accuracy metric.";
  }
    
  // LOG(INFO) << "Accuracy: " << accuracy;
  top[0]->mutable_cpu_data()[0] = accuracy / valid_pixel_count;
  // Accuracy layer should not be used as a loss function.
}

INSTANTIATE_CLASS(SegAccuracyLayer);
REGISTER_LAYER_CLASS(SEG_ACCURACY, SegAccuracyLayer);
}  // namespace caffe
