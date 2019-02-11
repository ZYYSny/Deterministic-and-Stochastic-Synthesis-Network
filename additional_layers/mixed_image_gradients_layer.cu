#include <cfloat>
#include <vector>

#include "caffe/layers/mixed_image_gradients_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MixedImageGradientsForward(int num, int chs, int height, int width, const Dtype threshold,
	const Dtype* left_data, const Dtype* right_data, const Dtype* index, Dtype* top) 
{
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;

  int img_idx = thread_idx / (chs * width * height);
  int img_res = thread_idx % (chs * width * height);
  int chs_idx = img_res / (width * height);
  int chs_res = img_res % (width * height);
  int h_idx   = chs_res / width;
  int w_idx   = chs_res % width;

  if(img_idx >= num || chs_idx >= chs || h_idx >= height || w_idx >= width) 
	  return;

  Dtype weight = index[img_idx*height*width+h_idx*width+w_idx];
  if(weight > threshold){
    top[thread_idx] = left_data[thread_idx];
  }else{
    top[thread_idx] = right_data[thread_idx];
  }
}

template <typename Dtype>
void MixedImageGradientsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  MixedImageGradientsForward<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
      bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(), Dtype(0.0),
      bottom[0]->gpu_data(), bottom[1]->gpu_data(), bottom[2]->gpu_data(), top[0]->mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
void MixedImageGradientsLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "Unimplemented.";
}

INSTANTIATE_LAYER_GPU_FUNCS(MixedImageGradientsLayer);

}  // namespace caffe