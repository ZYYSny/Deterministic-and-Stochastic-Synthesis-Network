#include <vector>

#include "caffe/layers/region_weighted_loss_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_functions.h>

#include <opencv2/opencv.hpp>
#include <chrono>

namespace caffe {

template <typename Dtype>
__global__ void RegionWeightedLossForward(int num, int chs, int height, int width, bool inverse, int pad,
	const Dtype* input, const Dtype* target, const Dtype* mask, Dtype* count, Dtype* loss) 
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

  bool hit = false;
  for(int i = -pad; i <= pad; i++){
    for(int j = -pad; j <= pad; j++){
	  int h = h_idx + i;
	  int w = w_idx + j;
	  if(h < 0) h = 0; if(h > (height-1)) h = height-1;
	  if(w < 0) w = 0; if(w > (width-1)) w = width-1;
	  if(mask[img_idx*height*width+h*width+w]>0){
	    hit = true;
	  }
	}
  }
  if(inverse){
    if(!hit){
	  loss[thread_idx] = abs(input[thread_idx]-target[thread_idx]);
	  count[thread_idx] = 1;
	}
  }else{
    if(hit){
	  loss[thread_idx] = abs(input[thread_idx]-target[thread_idx]);
	  count[thread_idx] = 1;	  
	}
  }
}

template <typename Dtype>
void RegionWeightedLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  loss_.scale_data(0.0);
  count_.scale_data(0.0);
  RegionWeightedLossForward<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
      bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),inverse_,pad_,
      bottom[0]->gpu_data(), bottom[1]->gpu_data(), bottom[2]->gpu_data(), 
	  count_.mutable_gpu_data(), loss_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
  Dtype loss = loss_.asum_data() / (1e-10+count_.asum_data());
  top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
__global__ void RegionWeightedLossBackward(int num, int chs, int height, int width, bool inverse, int pad,
	const Dtype* input, const Dtype* target, const Dtype* mask, Dtype* input_diff, Dtype* target_diff) 
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

  bool hit = false;
  for(int i = -pad; i <= pad; i++){
    for(int j = -pad; j <= pad; j++){
	  int h = h_idx + i;
	  int w = w_idx + j;
	  if(h < 0) h = 0; if(h > (height-1)) h = height-1;
	  if(w < 0) w = 0; if(w > (width-1)) w = width-1;
	  if(mask[img_idx*height*width+h*width+w]>0){
	    hit = true;
	  }
	}
  }
  if(inverse){
    if(!hit){
	  if(input[thread_idx]>target[thread_idx]){
	    input_diff[thread_idx] = 1;
		target_diff[thread_idx] = -1;
	  }else if(input[thread_idx]<target[thread_idx]){
	    input_diff[thread_idx] = -1;
		target_diff[thread_idx] = 1;	  
	  }else{
	    input_diff[thread_idx] = 0;
		target_diff[thread_idx] = 0;
	  }
	}
  }else{
    if(hit){
	  if(input[thread_idx]>target[thread_idx]){
	    input_diff[thread_idx] = 1;
		target_diff[thread_idx] = -1;
	  }else if(input[thread_idx]<target[thread_idx]){
	    input_diff[thread_idx] = -1;
		target_diff[thread_idx] = 1;	  
	  }else{
	    input_diff[thread_idx] = 0;
		target_diff[thread_idx] = 0;
	  }	  
	}
  }
}

template <typename Dtype>
void RegionWeightedLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  bottom[0]->scale_diff(0.0);
  bottom[1]->scale_diff(0.0);
  RegionWeightedLossBackward<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
      bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),inverse_,pad_,
      bottom[0]->gpu_data(), bottom[1]->gpu_data(), bottom[2]->gpu_data(), 
	  bottom[0]->mutable_gpu_diff(), bottom[1]->mutable_gpu_diff());
  CUDA_POST_KERNEL_CHECK;
  Dtype scale = top[0]->cpu_diff()[0] / (1e-10+count_.asum_data());
  bottom[0]->scale_diff(scale);
  bottom[1]->scale_diff(scale);
}

INSTANTIATE_LAYER_GPU_FUNCS(RegionWeightedLossLayer);

}  // namespace caffe