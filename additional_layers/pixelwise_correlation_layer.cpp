#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pixelwise_correlation_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PixelwiseCorrelationLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  PixelwiseGrammatrixParameter pixelwise_grammatrix_param = 
      this->layer_param_.pixelwise_grammatrix_param();
  CHECK_GT(pixelwise_grammatrix_param.kernel_size_size(), 0);
  total_channels_ = 0;
  for (int c = 0; c < pixelwise_grammatrix_param.kernel_size_size(); ++c) {
	CHECK_GT(pixelwise_grammatrix_param.kernel_size(c), 0);
    kernel_sizes_.push_back(pixelwise_grammatrix_param.kernel_size(c));
	CHECK_EQ(kernel_sizes_[c]%2,1);
	total_channels_ += kernel_sizes_[c]*kernel_sizes_[c];
  }
  total_channels_ *= bottom[0]->channels();
}

template <typename Dtype>
void PixelwiseCorrelationLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape({bottom[0]->num(),total_channels_,
      bottom[0]->height(),bottom[0]->width()});
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PixelwiseCorrelationLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LOG(FATAL) << "Not implemented!";
}

template <typename Dtype>
void PixelwiseCorrelationLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "Not implemented!";
}


#ifdef CPU_ONLY
STUB_GPU(PixelwiseCorrelationLayer);
#endif

INSTANTIATE_CLASS(PixelwiseCorrelationLayer);
REGISTER_LAYER_CLASS(PixelwiseCorrelation);

}  // namespace caffe