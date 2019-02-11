#include <vector>

#include "caffe/layers/region_weighted_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void RegionWeightedLossLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->num(), bottom[2]->num());
  CHECK_EQ(1, bottom[2]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[2]->height());
  CHECK_EQ(bottom[0]->width(), bottom[2]->width()); 
  kernel_size_= this->layer_param_.region_weighted_loss_param().kernel_size();
  inverse_= this->layer_param_.region_weighted_loss_param().inverse();
  pad_ = kernel_size_/2;
  CHECK_EQ(kernel_size_%2,1);
}

template <typename Dtype>
void RegionWeightedLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->count(1), bottom[1]->count(1))
      << "Inputs must have the same dimension.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->num(), bottom[2]->num());
  CHECK_EQ(1, bottom[2]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[2]->height());
  CHECK_EQ(bottom[0]->width(), bottom[2]->width()); 
  count_.ReshapeLike(*bottom[0]);
  loss_.ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void RegionWeightedLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LOG(FATAL) << "Not Implemented.";
}

template <typename Dtype>
void RegionWeightedLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "Not Implemented.";
}

#ifdef CPU_ONLY
STUB_GPU(RegionWeightedLossLayer);
#endif

INSTANTIATE_CLASS(RegionWeightedLossLayer);
REGISTER_LAYER_CLASS(RegionWeightedLoss);

}  // namespace caffe