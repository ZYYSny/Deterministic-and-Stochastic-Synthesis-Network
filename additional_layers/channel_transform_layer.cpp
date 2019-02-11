#include <vector>
#include <math.h>
#include <sstream>
#include <string>

#include "caffe/layers/channel_transform_layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void ChannelTransformLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ChannelTransformParameter channel_transform_param = 
      this->layer_param_.channel_transform_param();
  CHECK_EQ(channel_transform_param.mean_size(), bottom[0]->channels());
  CHECK_EQ(channel_transform_param.scale_size(), bottom[0]->channels());
  for (int c = 0; c < bottom[0]->channels(); ++c) {
    means_.push_back(channel_transform_param.mean(c));
	  scales_.push_back(channel_transform_param.scale(c));
  }
}

template <typename Dtype>
void ChannelTransformLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->ReshapeLike(*bottom[0]);
}

template <typename Dtype>
void ChannelTransformLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  //LOG(INFO)<<top[0]->num()<<"\t"<<top[0]->channels()<<"\t"<<top[0]->height()<<"\t"<<top[0]->width();
  const Dtype *bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const int num = bottom[0]->num();
  const int chs = bottom[0]->channels();
  const int dim = bottom[0]->height()*bottom[0]->width();
  
  caffe_copy(bottom[0]->count(), bottom_data, top_data);
  //LOG(INFO)<<top[0]->num()<<"\t"<<top[0]->channels()<<"\t"<<top[0]->height()<<"\t"<<top[0]->width();
  for(int n = 0; n < num; n++){
	for(int c = 0; c < chs; c++){
	  caffe_gpu_add_scalar(dim, Dtype(-1*means_[c]), top_data + n * chs * dim + c * dim);
	  caffe_gpu_scale(dim, Dtype(1/Dtype(scales_[c])), 
	      top_data + n * chs * dim + c * dim,
          top_data + n * chs * dim + c * dim);
	}
  }
  //LOG(INFO)<<top[0]->num()<<"\t"<<top[0]->channels()<<"\t"<<top[0]->height()<<"\t"<<top[0]->width();
  /*if(this->layer_param_.preserve_cpu_data()){
	 const Dtype* bottom_cpu_data = bottom[0]->cpu_data();
     bottom[0]->release_gpu_data();
  }else{
     bottom[0]->release_gpu_data();
  }*/
  //const Dtype* top_cpu_data = top[0]->cpu_data();
  //top[0]->release_gpu_data();
}

template <typename Dtype>
void ChannelTransformLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();
  const int num = bottom[0]->num();
  const int chs = bottom[0]->channels();
  const int dim = bottom[0]->height()*bottom[0]->width();
  
  caffe_copy(bottom[0]->count(), top_diff, bottom_diff);
  
  for(int n = 0; n < num; n++){
	for(int c = 0; c < chs; c++){
	  caffe_gpu_scale(dim, Dtype(1/Dtype(scales_[c])), 
	      top_diff    + n * chs * dim + c * dim,
          bottom_diff + n * chs * dim + c * dim);
	}
  }
}

INSTANTIATE_CLASS(ChannelTransformLayer);
REGISTER_LAYER_CLASS(ChannelTransform);

}  // namespace caffe
