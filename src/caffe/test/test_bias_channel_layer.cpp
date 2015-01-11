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
class BiasChannelLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BiasChannelLayerTest()
      : blob_bottom_0(new Blob<Dtype>(2, 4, 3, 2)),
        blob_bottom_1(new Blob<Dtype>(2, 2, 1, 1)),
        blob_top_(new Blob<Dtype>()) {}
  virtual void SetUp() {
    // fill the values
    shared_ptr<GaussianFiller<Dtype> > filler;
    FillerParameter filler_param;
    filler.reset(new GaussianFiller<Dtype>(filler_param));
    filler->Fill(this->blob_bottom_0);
    for (int i = 0; i < blob_bottom_1->count(); ++i) {
      blob_bottom_1->mutable_cpu_data()[i] = 1 + caffe_rng_rand() % 3;  // 1, 2, 3
    }
    blob_bottom_vec_.push_back(blob_bottom_0);
    blob_bottom_vec_.push_back(blob_bottom_1);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~BiasChannelLayerTest() {
    delete blob_bottom_0; delete blob_bottom_1;
    delete blob_top_;
  }

  Blob<Dtype>* const blob_bottom_0;
  Blob<Dtype>* const blob_bottom_1;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BiasChannelLayerTest, TestDtypesAndDevices);

TYPED_TEST(BiasChannelLayerTest, TestSetup) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  layer_param.mutable_bias_channel_param()->set_bg_bias(1.);
  layer_param.mutable_bias_channel_param()->set_fg_bias(2.);
  BiasChannelLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_->num(), this->blob_bottom_0->num());
  EXPECT_EQ(this->blob_top_->channels(), this->blob_bottom_0->channels());
  EXPECT_EQ(this->blob_top_->height(), this->blob_bottom_0->height());
  EXPECT_EQ(this->blob_top_->width(), this->blob_bottom_0->width());
}

TYPED_TEST(BiasChannelLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BiasChannelLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0);
}

}  // namespace caffe
