#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/gradient_operator_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void GradientOperatorLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  kernel_size_ = this->layer_param_.gradient_operator_param().kernel_size();
  CHECK_EQ(1, kernel_size_%2);
  aug_chs_ = kernel_size_ * kernel_size_ - 1;
  total_chs_ = aug_chs_ * bottom[0]->channels();
}

template <typename Dtype>
void GradientOperatorLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->Reshape({bottom[0]->num(), total_chs_, bottom[0]->height(), bottom[0]->width()});
}

template <typename Dtype>
void GradientOperatorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  LOG(FATAL) << "Not Implemented.";
}

template <typename Dtype>
void GradientOperatorLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "Not Implemented.";
}

#ifdef CPU_ONLY
STUB_GPU(GradientOperatorLayer);
#endif

INSTANTIATE_CLASS(GradientOperatorLayer);
REGISTER_LAYER_CLASS(GradientOperator);

}  // namespace caffe