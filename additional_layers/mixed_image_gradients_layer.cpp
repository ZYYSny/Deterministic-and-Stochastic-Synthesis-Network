#include <cfloat>
#include <vector>

#include "caffe/layers/mixed_image_gradients_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void MixedImageGradientsLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 3);
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->num(), bottom[2]->num());
  CHECK_EQ(1, bottom[2]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[2]->height());
  CHECK_EQ(bottom[0]->width(), bottom[2]->width());  
}

template <typename Dtype>
void MixedImageGradientsLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->num(), bottom[2]->num());
  CHECK_EQ(1, bottom[2]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[2]->height());
  CHECK_EQ(bottom[0]->width(), bottom[2]->width()); 
  top[0]->Reshape({bottom[0]->num(),bottom[0]->channels(),bottom[0]->height(),bottom[0]->width()});
}

template <typename Dtype>
void MixedImageGradientsLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LOG(FATAL) << "Unimplemented.";
}

template <typename Dtype>
void MixedImageGradientsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "Unimplemented.";
}

#ifdef CPU_ONLY
STUB_GPU(MixedImageGradientsLayer);
#endif

INSTANTIATE_CLASS(MixedImageGradientsLayer);
REGISTER_LAYER_CLASS(MixedImageGradients);

}  // namespace caffe
