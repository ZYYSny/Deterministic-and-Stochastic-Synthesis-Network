#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/gradient_operator_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_functions.h>

#include <opencv2/opencv.hpp>
#include <chrono>

namespace caffe {

template <typename Dtype>
__global__ void GradientOperatorForward(int num, int chs, int height, int width,
    int total_chs, int aug_chs, int kernel_size, const Dtype* bottom_data, Dtype* top_data) 
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

  int aug_chs_idx = 0;
  int half_kernel = kernel_size / 2;
  for(int m = -half_kernel; m <= half_kernel; m++){
    for(int n = -half_kernel; n <= half_kernel; n++){
      if(m == 0 || n == 0) continue;
	  int diff_h_idx = h_idx + m;
	  int diff_w_idx = w_idx + n;
	  if(diff_h_idx < 0) diff_h_idx = 0;
	  if(diff_h_idx >= height) diff_h_idx = height - 1;
	  if(diff_w_idx < 0) diff_w_idx = 0;
	  if(diff_w_idx >= width) diff_w_idx = width - 1;
	  int diff_chs_idx = chs_idx * aug_chs + aug_chs_idx;
	  top_data[img_idx*total_chs*height*width+diff_chs_idx*height*width+h_idx*width+w_idx] = 
	      bottom_data[img_idx*chs*height*width+chs_idx*height*width+diff_h_idx*width+diff_w_idx] - 
		  bottom_data[img_idx*chs*height*width+chs_idx*height*width+h_idx*width+w_idx];
	  aug_chs_idx++;
	  if(aug_chs_idx > aug_chs) printf("Error in channel idx caculation!");
    }
  }
}

template <typename Dtype>
void GradientOperatorLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  top[0]->scale_data(0);
  GradientOperatorForward<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
      bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),
      total_chs_, aug_chs_, kernel_size_, bottom[0]->gpu_data(), top[0]->mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void GradientOperatorBackward(int num, int chs, int height, int width,
    int total_chs, int aug_chs, int kernel_size, Dtype* bottom_diff, const Dtype* top_diff) 
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

  int aug_chs_idx = 0;
  int half_kernel = kernel_size / 2;
  for(int m = -half_kernel; m <= half_kernel; m++){
    for(int n = -half_kernel; n <= half_kernel; n++){
      if(m == 0 || n == 0) continue;
	  int diff_h_idx = h_idx + m;
	  int diff_w_idx = w_idx + n;
	  if(diff_h_idx < 0) diff_h_idx = 0;
	  if(diff_h_idx >= height) diff_h_idx = height - 1;
	  if(diff_w_idx < 0) diff_w_idx = 0;
	  if(diff_w_idx >= width) diff_w_idx = width - 1;
	  int diff_chs_idx = chs_idx * aug_chs + aug_chs_idx;
	  Dtype top_loc_diff = 
	      top_diff[img_idx*total_chs*height*width+diff_chs_idx*height*width+h_idx*width+w_idx];
	  caffe_gpu_atomic_add(Dtype(top_loc_diff), 
	      bottom_diff+img_idx*chs*height*width+chs_idx*height*width+diff_h_idx*width+diff_w_idx);
	  caffe_gpu_atomic_add(Dtype(-top_loc_diff), 
	      bottom_diff+img_idx*chs*height*width+chs_idx*height*width+h_idx*width+w_idx);
	  aug_chs_idx++;
	  if(aug_chs_idx > aug_chs) printf("Error in channel idx caculation!");
    }
  }
}

template <typename Dtype>
void GradientOperatorLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  bottom[0]->scale_diff(0);
  GradientOperatorBackward<<<CAFFE_GET_BLOCKS(bottom[0]->count()), CAFFE_CUDA_NUM_THREADS>>>(
      bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width(),
      total_chs_, aug_chs_, kernel_size_, bottom[0]->mutable_gpu_diff(), top[0]->gpu_diff());
  CUDA_POST_KERNEL_CHECK;
}

INSTANTIATE_LAYER_GPU_FUNCS(GradientOperatorLayer);

}  // namespace caffe
