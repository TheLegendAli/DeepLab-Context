#include <vector>

#include "caffe/layer.hpp"
#include "caffe/neuron_layers.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/filler.hpp"

#include <algorithm>

namespace caffe {

template <typename Dtype>
void GainChannelLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  GainChannelParameter param = this->layer_param_.gain_channel_param();
  channels_ = bottom[0]->channels();
  num_output_nz_ = param.num_output_nz();
  drift_ = param.drift();
  stdev_ = param.stdev();
  norm_mean_ = param.norm_mean();
  CHECK(num_output_nz_ > 0 && num_output_nz_ <= channels_);
  CHECK_GE(drift_, 0) << "Drift needs to be non-negative";
  CHECK_GE(stdev_, 0) << "Noise stdev needs to be non-negative";
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
    CHECK_EQ(this->blobs_[0]->count(), channels_);
  } else {
    CHECK(param.has_gain_filler())
      << "Need to specify the gain filler (typically const = 1)";
    this->blobs_.resize(1);
    // Initialize and fill the weights
    this->blobs_[0].reset(new Blob<Dtype>(1, channels_, 1, 1));
    shared_ptr<Filler<Dtype> > gain_filler(GetFiller<Dtype>(
        param.gain_filler()));
    gain_filler->Fill(this->blobs_[0].get());
  }
}

template <typename Dtype>
void GainChannelLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  num_ = bottom[0]->num();
  CHECK_EQ(bottom[0]->channels(), channels_);
  height_ = bottom[0]->height();
  width_ = bottom[0]->width();
  //
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void GainChannelLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Project gain to positive values
  Dtype *gain_data = this->blobs_[0]->mutable_cpu_data();
  for (int c = 0; c < channels_; ++c) {
    if (gain_data[c] < 0) {
      gain_data[c] = Dtype(0);
    }
  }
  // Optionally (multiplicatively) normalize gains to be 1-mean
  if (norm_mean_) {
    Dtype sum = 0;
    for (int c = 0; c < channels_; ++c) {
      sum += gain_data[c];
    }
    for (int c = 0; c < channels_; ++c) {
      gain_data[c] = (sum > 0) ? channels_ / sum * gain_data[c] : Dtype(1);
    }
  }
  // Multiply with gain
  for (int n = 0; n < num_; ++n) {
    for (int c = 0; c < channels_; ++c) {
      caffe_cpu_scale(height_ * width_, gain_data[c],
		      bottom[0]->cpu_data(n, c),
		      top[0]->mutable_cpu_data(n, c));
    }
  }
}

template <typename Dtype>
void GainChannelLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const bool in_place = bottom[0]->cpu_data() == top[0]->cpu_data();
  // Update local parameters
  if (1 || this->param_propagate_down(0)) {
    CHECK_EQ(this->blobs_[0]->count(), channels_);
    Dtype *gain_diff = this->blobs_[0]->mutable_cpu_diff();
    caffe_set(channels_, Dtype(0), gain_diff);
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
	const Dtype alpha = this->blobs_[0]->cpu_data()[c];
	gain_diff[c] +=
	  ((in_place && alpha > 0) ? Dtype(1.) / alpha : Dtype(1.)) *
	  caffe_cpu_dot(height_ * width_,
	     top[0]->cpu_diff(n, c),
	     bottom[0]->cpu_data(n, c));
      }
    }
    if (drift_ > 0) {
      // Find the num_output_nz_ greatest gains
      const Dtype *gain_data = this->blobs_[0]->cpu_data();
      std::vector<Dtype> gains(channels_);
      std::copy(gain_data, gain_data + channels_, gains.begin());
      std::nth_element(gains.begin(), gains.begin() + num_output_nz_,
		       gains.end(), std::greater<Dtype>());
      // Add gradient drift to encourage further shrinkage of small gains
      for (int c = 0; c < channels_; ++c) {
	if (gain_data[c] < gains[num_output_nz_ - 1]) {
	  gain_diff[c] += drift_;
	}
      }
    }
    if (stdev_ > 0) {
      // Add zero-mean Gaussian noise to the gradient
      std::vector<Dtype> noise(channels_);
      caffe_rng_gaussian(channels_, Dtype(0),
	 Dtype(1.0), &noise[0]);
      caffe_axpy(channels_, stdev_, &noise[0], gain_diff);
    }
    if (norm_mean_) {
      // Make the gradient zero-mean
      Dtype sum = 0;
      for (int c = 0; c < channels_; ++c) {
	sum += gain_diff[c];
      }
      sum /= channels_;
      for (int c = 0; c < channels_; ++c) {
	gain_diff[c] -= sum;
      }      
    }
  }
  if (propagate_down[0]) {
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
	const Dtype alpha = this->blobs_[0]->cpu_data()[c];
	caffe_cpu_scale(height_ * width_, alpha,
			top[0]->cpu_diff(n, c),
			bottom[0]->mutable_cpu_diff(n, c));
      }
    }
  }
}

template <typename Dtype>
void GainChannelLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Project gain to positive values
  Dtype *gain_data = this->blobs_[0]->mutable_cpu_data();
  for (int c = 0; c < channels_; ++c) {
    if (gain_data[c] < 0) {
      gain_data[c] = Dtype(0);
    }
  }
  // Optionally (multiplicatively) normalize gains to be 1-mean
  if (norm_mean_) {
    Dtype sum = 0;
    for (int c = 0; c < channels_; ++c) {
      sum += gain_data[c];
    }
    for (int c = 0; c < channels_; ++c) {
      gain_data[c] = (sum > 0) ? channels_ / sum * gain_data[c] : Dtype(1);
    }
  }
  // Multiply with gain
  for (int n = 0; n < num_; ++n) {
    for (int c = 0; c < channels_; ++c) {
      caffe_gpu_scale(height_ * width_, gain_data[c],
		      bottom[0]->gpu_data(n, c),
		      top[0]->mutable_gpu_data(n, c));
    }
  }
}

template <typename Dtype>
void GainChannelLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const bool in_place = bottom[0]->gpu_data() == top[0]->gpu_data();
  // Update local parameters
  if (1 || this->param_propagate_down(0)) {
    CHECK_EQ(this->blobs_[0]->count(), channels_);
    Dtype *gain_diff = this->blobs_[0]->mutable_gpu_diff();
    caffe_gpu_set(channels_, Dtype(0), gain_diff);
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
	const Dtype alpha = this->blobs_[0]->cpu_data()[c];
	caffe_gpu_gemm(CblasNoTrans, CblasNoTrans,
	   1, 1, height_ * width_,
	   ((in_place && alpha > 0) ? Dtype(1.) / alpha : Dtype(1.)),
	   top[0]->gpu_diff(n, c), bottom[0]->gpu_data(n, c),
	   Dtype(1.), &gain_diff[c]);
      }
    }
    gain_diff = this->blobs_[0]->mutable_cpu_diff();
    if (drift_ > 0) {
      // Find the num_output_nz_ greatest gains
      const Dtype *gain_data = this->blobs_[0]->cpu_data();
      std::vector<Dtype> gains(channels_);
      std::copy(gain_data, gain_data + channels_, gains.begin());
      std::nth_element(gains.begin(), gains.begin() + num_output_nz_,
		       gains.end(), std::greater<Dtype>());
      // Add gradient drift to encourage further shrinkage of small gains
      for (int c = 0; c < channels_; ++c) {
	if (gain_data[c] < gains[num_output_nz_ - 1]) {
	  gain_diff[c] += drift_;
	}
      }
    }
    if (stdev_ > 0) {
      // Add zero-mean Gaussian noise to the gradient
      std::vector<Dtype> noise(channels_);
      caffe_rng_gaussian(channels_, Dtype(0),
	 Dtype(1.0), &noise[0]);
      caffe_axpy(channels_, stdev_, &noise[0], gain_diff);
    }
    if (norm_mean_) {
      // Make the gradient zero-mean
      Dtype sum = 0;
      for (int c = 0; c < channels_; ++c) {
	sum += gain_diff[c];
      }
      sum /= channels_;
      for (int c = 0; c < channels_; ++c) {
	gain_diff[c] -= sum;
      }      
    }
  }
  if (propagate_down[0]) {
    for (int n = 0; n < num_; ++n) {
      for (int c = 0; c < channels_; ++c) {
	const Dtype alpha = this->blobs_[0]->cpu_data()[c];
	caffe_gpu_scale(height_ * width_, alpha,
			top[0]->gpu_diff(n, c),
			bottom[0]->mutable_gpu_diff(n, c));
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(GainChannelLayer);
#endif

INSTANTIATE_CLASS(GainChannelLayer);
REGISTER_LAYER_CLASS(GAIN_CHANNEL, GainChannelLayer);

}  // namespace caffe
