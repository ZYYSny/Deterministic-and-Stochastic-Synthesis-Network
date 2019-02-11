#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pixelwise_grammatrix_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PixelwiseGrammatrixLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
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
  CHECK_EQ(bottom[0]->num(),bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(),bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(),bottom[1]->height());
  CHECK_EQ(bottom[0]->width(),bottom[1]->width());
}

template <typename Dtype>
void PixelwiseGrammatrixLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  diff_.ReshapeLike(*bottom[0]);
  loss_.ReshapeLike(*bottom[0]);
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PixelwiseGrammatrixLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LOG(FATAL) << "Not implemented!";
}

template <typename Dtype>
void PixelwiseGrammatrixLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "Not implemented!";
}


#ifdef CPU_ONLY
STUB_GPU(PixelwiseGrammatrixLayer);
#endif

INSTANTIATE_CLASS(PixelwiseGrammatrixLayer);
REGISTER_LAYER_CLASS(PixelwiseGrammatrix);

}  // namespace caffe