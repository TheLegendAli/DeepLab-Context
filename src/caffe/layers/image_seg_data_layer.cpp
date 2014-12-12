#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/data_layers.hpp"
#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
ImageSegDataLayer<Dtype>::~ImageSegDataLayer<Dtype>() {
  this->JoinPrefetchThread();
}

template <typename Dtype>
void ImageSegDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  string root_folder = this->layer_param_.image_data_param().root_folder();

  TransformationParameter transform_param = this->layer_param_.transform_param();
  CHECK(transform_param.has_mean_file() == false) << 
         "ImageSegDataLayer does not support mean file";

  const int max_expected_channel = this->layer_param_.image_data_param().max_expected_channel();
  const int max_expected_height = this->layer_param_.image_data_param().max_expected_height();
  const int max_expected_width = this->layer_param_.image_data_param().max_expected_width();

  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string imgfn;
  string segfn;
  while (infile >> imgfn >> segfn) {
    lines_.push_back(std::make_pair(imgfn, segfn));
  }

  // TODO: check this
  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";

  lines_id_ = 0;
  // TODO: check this. Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  int channels;
  int height;
  int width;

  if (max_expected_height == 0 && max_expected_width == 0) {
    // Read an image, and use it to initialize the top blob.
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);

    channels = cv_img.channels();
    height = cv_img.rows;
    width = cv_img.cols;
  } else {
    channels = max_expected_channel;
    height = max_expected_height;
    width  = max_expected_width;
  }

  // image
  const int crop_size = this->layer_param_.transform_param().crop_size();
  const int batch_size = this->layer_param_.image_data_param().batch_size();

  if (crop_size > 0) {
    top[0]->Reshape(batch_size, channels, crop_size, crop_size);
    this->prefetch_data_.Reshape(batch_size, channels, crop_size, crop_size);
    this->transformed_data_.Reshape(1, channels, crop_size, crop_size);

    //label
    top[1]->Reshape(batch_size, 1, crop_size, crop_size);
    this->prefetch_label_.Reshape(batch_size, 1, crop_size, crop_size);
    this->transformed_label_.Reshape(1, 1, crop_size, crop_size);
  } else {
    top[0]->Reshape(batch_size, channels, height, width);
    this->prefetch_data_.Reshape(batch_size, channels, height, width);
    this->transformed_data_.Reshape(1, channels, height, width);

    //label
    top[1]->Reshape(batch_size, 1, height, width);
    this->prefetch_label_.Reshape(batch_size, 1, height, width);
    this->transformed_label_.Reshape(1, 1, height, width);
  }
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  LOG(INFO) << "output label size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width();
}

template <typename Dtype>
void ImageSegDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is used to create a thread that prefetches the data.
template <typename Dtype>
void ImageSegDataLayer<Dtype>::InternalThreadEntry() {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(this->prefetch_data_.count());
  CHECK(this->transformed_data_.count());
  Dtype* top_data = this->prefetch_data_.mutable_cpu_data();
  Dtype* top_label = this->prefetch_label_.mutable_cpu_data(); 
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  TransformationParameter transorm_param = this->layer_param_.transform_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  const int lines_size = lines_.size();

  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
				      new_height, new_width, is_color);
    if (!cv_img.data) {
      DLOG(INFO) << "Fail to load img: " << root_folder + lines_[lines_id_].first;
    }

    // check input dimensions
    //int channels = cv_img.channels();
    //int height = cv_img.rows;
    //int width = cv_img.cols;

    /*
    int pad_height = this->prefetch_data_.height() - height;
    int pad_width  = this->prefetch_data_.width() - width;
    
    if (pad_height > 0 || pad_width > 0) {
      Dtype b_val = transorm_param.mean_value(0);
      Dtype g_val = transorm_param.mean_value(1);
      Dtype r_val = transorm_param.mean_value(2);

      cv::copyMakeBorder(cv_img, cv_img, 0, pad_height, 0, pad_width, cv::BORDER_CONSTANT, cv::Scalar(b_val, g_val, r_val));
      
      cv::namedWindow("img", cv::WINDOW_AUTOSIZE);
      cv::imshow("img", cv_img);
      cv::waitKey(0);
      
    }
    */
    /*
      this->prefetch_data_.Reshape(1, channels, height, width);
      this->transformed_data_.Reshape(1, channels, height, width);
      this->prefetch_label_.Reshape(1, 1, height, width);
      this->transformed_label_.Reshape(1, 1, height, width); 
    */

    read_time += timer.MicroSeconds();
    timer.Start();
    // Apply transformations (mirror, crop...) to the image
    int offset;
    offset = this->prefetch_data_.offset(item_id);
    this->transformed_data_.set_cpu_data(top_data + offset);
    this->data_transformer_.TransformAndPad(cv_img, &(this->transformed_data_));
    trans_time += timer.MicroSeconds();


    /* // debug
    std::string save_fn;
    save_fn = "img.bin";
    this->transformed_data_.WriteToBinaryFile(save_fn);
    // */

    timer.Start();
    cv::Mat cv_seg = ReadImageToCVMat(root_folder + lines_[lines_id_].second,
				      new_height, new_width, false);
    if (!cv_seg.data) {
      DLOG(INFO) << "Fail to load img: " << root_folder + lines_[lines_id_].second;
    }
  
    /*
    if (pad_height > 0 || pad_width > 0) {
      Dtype mask_val = 255;

      cv::copyMakeBorder(cv_seg, cv_seg, 0, pad_height, 0, pad_width, cv::BORDER_CONSTANT, cv::Scalar(mask_val));
      
      cv::namedWindow("seg", cv::WINDOW_AUTOSIZE);
      cv::imshow("seg", cv_seg);
      cv::waitKey(0);      
    }
    */

    /* //jay debug
    for(int m = 0; m < cv_seg.rows; ++m) {
       for(int n = 0; n < cv_seg.cols; ++n) {
	 std::cout << (double)cv_seg.at<uchar>(m,n) << ' ';
       }
       std::cout << std::endl;
    }
    std::cout << std::endl;
    // */

    read_time += timer.MicroSeconds();
 
    timer.Start();
    offset = this->prefetch_label_.offset(item_id);
    this->transformed_label_.set_cpu_data(top_label + offset);
    this->data_transformer_.TransformSegAndPad(cv_seg, &(this->transformed_label_));
    trans_time += timer.MicroSeconds();

    /* // debug
    save_fn = "seg.bin";
    this->transformed_label_.WriteToBinaryFile(save_fn);
    // */


    // go to the next std::vector<int>::iterator iter;
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
	ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageSegDataLayer);
REGISTER_LAYER_CLASS(IMAGE_SEG_DATA, ImageSegDataLayer);
}  // namespace caffe
