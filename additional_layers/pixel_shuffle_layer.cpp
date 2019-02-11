#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pixel_shuffle_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PixelShuffleLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  scale_ = this->layer_param_.pixel_shuffle_param().scale();
  CHECK_EQ(bottom[0]->channels()%scale_,0);
  scale_sqrt_ = std::sqrt(scale_);
  CHECK_EQ(scale_sqrt_*scale_sqrt_,scale_);
  channels_ = bottom[0]->channels()/scale_;
  height_ = bottom[0]->height()*scale_sqrt_;
  width_ = bottom[0]->width()*scale_sqrt_;
}

template <typename Dtype>
void PixelShuffleLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  scale_ = this->layer_param_.pixel_shuffle_param().scale();
  CHECK_EQ(bottom[0]->channels()%scale_,0);
  scale_sqrt_ = std::sqrt(scale_);
  CHECK_EQ(scale_sqrt_*scale_sqrt_,scale_);
  channels_ = bottom[0]->channels()/scale_;
  height_ = bottom[0]->height()*scale_sqrt_;
  width_ = bottom[0]->width()*scale_sqrt_;
  top[0]->Reshape({bottom[0]->num(),channels_,height_,width_});
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PixelShuffleLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  LOG(FATAL) << "Not implemented!";
}

template <typename Dtype>
void PixelShuffleLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "Not implemented!";
}


#ifdef CPU_ONLY
STUB_GPU(PixelShuffleLayer);
#endif

INSTANTIATE_CLASS(PixelShuffleLayer);
REGISTER_LAYER_CLASS(PixelShuffle);

}  // namespace caffe
