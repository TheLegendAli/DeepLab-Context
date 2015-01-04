#include <utility>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

template <typename Dtype>
class ArgMaxLayerTest;

template <typename Dtype>
void test_fun(ArgMaxLayerTest<Dtype> *test,
	      const bool out_max_val, const int top_k);

template <typename Dtype>
class ArgMaxLayerTest : public ::testing::Test {
 protected:
  ArgMaxLayerTest()
      : blob_bottom_(new Blob<Dtype>(10, 20, 3, 4)),
        blob_top_(new Blob<Dtype>()),
        top_k_(5) {
    Caffe::set_mode(Caffe::CPU);
    Caffe::set_random_seed(1701);
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~ArgMaxLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
  size_t top_k_;

  friend void test_fun<Dtype>(ArgMaxLayerTest<Dtype> *test,
	const bool out_max_val, const int top_k);
};

TYPED_TEST_CASE(ArgMaxLayerTest, TestDtypes);

TYPED_TEST(ArgMaxLayerTest, TestSetup) {
  LayerParameter layer_param;
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), 1);
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(ArgMaxLayerTest, TestSetupMaxVal) {
  LayerParameter layer_param;
  ArgMaxParameter* argmax_param = layer_param.mutable_argmax_param();
  argmax_param->set_out_max_val(true);
  ArgMaxLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), 2);
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

template <typename Dtype>
void test_fun(ArgMaxLayerTest<Dtype> *test,
	      const bool out_max_val, const int top_k) {
  LayerParameter layer_param;
  layer_param.mutable_argmax_param()->set_out_max_val(out_max_val);
  layer_param.mutable_argmax_param()->set_top_k(top_k);
  ArgMaxLayer<Dtype> layer(layer_param);
  layer.SetUp(test->blob_bottom_vec_, test->blob_top_vec_);
  layer.Forward(test->blob_bottom_vec_, test->blob_top_vec_);
  const int num = test->blob_bottom_->num();
  const int channels = test->blob_bottom_->channels();
  const int height = test->blob_bottom_->height();
  const int width = test->blob_bottom_->width();
  const int channel_offset = height * width;
  for (int n = 0; n < num; ++n) {
    for (int h = 0; h < height; ++h) {
      for (int w = 0; w < width; ++w) {
	// Now, check values
	const Dtype* bottom_data = test->blob_bottom_->cpu_data(n, 0, h, w);
	const Dtype* top_data = test->blob_top_->cpu_data(n, 0, h, w);
	for (int k = 0; k < top_k; ++k) {
	  const int max_ind = static_cast<int>(top_data[k * channel_offset]);
	  EXPECT_GE(max_ind, 0);
	  EXPECT_LT(max_ind, channels);
	  const Dtype max_val = bottom_data[max_ind * channel_offset];
	  if (out_max_val) {
	    EXPECT_EQ(top_data[(top_k + k) * channel_offset], max_val);
	  }
	  int count = 0;
	  for (int c = 0; c < channels; ++c) {
	    if (bottom_data[c * channel_offset] > max_val) {
	      ++count;
	    }
	  }
	  EXPECT_EQ(k, count);
	}
      }
    }
  }
}

TYPED_TEST(ArgMaxLayerTest, TestCPU) {
  test_fun<TypeParam>(this, false, 1);
}

TYPED_TEST(ArgMaxLayerTest, TestCPUMaxVal) {
  test_fun<TypeParam>(this, true, 1);
}

TYPED_TEST(ArgMaxLayerTest, TestCPUTopK) {
  test_fun<TypeParam>(this, false, this->top_k_);
}

TYPED_TEST(ArgMaxLayerTest, TestCPUMaxValTopK) {
  test_fun<TypeParam>(this, true, this->top_k_);
}

}  // namespace caffe
