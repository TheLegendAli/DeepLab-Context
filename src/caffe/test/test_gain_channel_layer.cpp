#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class GainChannelLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  GainChannelLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    shared_ptr<GaussianFiller<Dtype> > filler;
    FillerParameter filler_param;
    filler.reset(new GaussianFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~GainChannelLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(GainChannelLayerTest, TestDtypesAndDevices);

TYPED_TEST(GainChannelLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_gain_channel_param()->set_num_output_nz(3);
  layer_param.mutable_gain_channel_param()->mutable_gain_filler()->set_type("constant");
  layer_param.mutable_gain_channel_param()->mutable_gain_filler()->set_value(1.f);
  GainChannelLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_->width());
}

TYPED_TEST(GainChannelLayerTest, TestCPU) {
  typedef typename TypeParam::Dtype Dtype;
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  layer_param.mutable_gain_channel_param()->set_num_output_nz(3);
  layer_param.mutable_gain_channel_param()->mutable_gain_filler()->set_type("constant");
  layer_param.mutable_gain_channel_param()->mutable_gain_filler()->set_value(1.f);
  layer_param.mutable_gain_channel_param()->set_norm_mean(true);
  GainChannelLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_bottom_->cpu_data()[i],
		this->blob_top_->cpu_data()[i], 1e-5);
  }
}

TYPED_TEST(GainChannelLayerTest, TestGPU) {
  typedef typename TypeParam::Dtype Dtype;
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  layer_param.mutable_gain_channel_param()->set_num_output_nz(3);
  layer_param.mutable_gain_channel_param()->mutable_gain_filler()->set_type("constant");
  layer_param.mutable_gain_channel_param()->mutable_gain_filler()->set_value(1.f);
  layer_param.mutable_gain_channel_param()->set_norm_mean(true);
  GainChannelLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  for (int i = 0; i < this->blob_bottom_->count(); ++i) {
    EXPECT_NEAR(this->blob_bottom_->cpu_data()[i],
		this->blob_top_->cpu_data()[i], 1e-5);
  }
}

TYPED_TEST(GainChannelLayerTest, TestGradientCPU) {
  typedef typename TypeParam::Dtype Dtype;
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  layer_param.mutable_gain_channel_param()->set_num_output_nz(3);
  layer_param.mutable_gain_channel_param()->mutable_gain_filler()->set_type("constant");
  layer_param.mutable_gain_channel_param()->mutable_gain_filler()->set_value(1.f);
  layer_param.mutable_gain_channel_param()->set_norm_mean(false);
  GainChannelLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(GainChannelLayerTest, TestGradientGPU) {
  typedef typename TypeParam::Dtype Dtype;
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  layer_param.mutable_gain_channel_param()->set_num_output_nz(3);
  layer_param.mutable_gain_channel_param()->mutable_gain_filler()->set_type("constant");
  layer_param.mutable_gain_channel_param()->mutable_gain_filler()->set_value(1.f);
  layer_param.mutable_gain_channel_param()->set_norm_mean(false);
  GainChannelLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(GainChannelLayerTest, TestGradientCPUNorm) {
  typedef typename TypeParam::Dtype Dtype;
  Caffe::set_mode(Caffe::CPU);
  LayerParameter layer_param;
  layer_param.mutable_gain_channel_param()->set_num_output_nz(3);
  layer_param.mutable_gain_channel_param()->mutable_gain_filler()->set_type("constant");
  layer_param.mutable_gain_channel_param()->mutable_gain_filler()->set_value(1.f);
  layer_param.mutable_gain_channel_param()->set_norm_mean(true);
  GainChannelLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(GainChannelLayerTest, TestGradientGPUNorm) {
  typedef typename TypeParam::Dtype Dtype;
  Caffe::set_mode(Caffe::GPU);
  LayerParameter layer_param;
  layer_param.mutable_gain_channel_param()->set_num_output_nz(3);
  layer_param.mutable_gain_channel_param()->mutable_gain_filler()->set_type("constant");
  layer_param.mutable_gain_channel_param()->mutable_gain_filler()->set_value(1.f);
  layer_param.mutable_gain_channel_param()->set_norm_mean(true);
  GainChannelLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
