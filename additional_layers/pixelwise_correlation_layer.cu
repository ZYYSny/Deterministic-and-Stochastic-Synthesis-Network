#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pixelwise_correlation_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_functions.h>

namespace caffe {

// one thread for every pixel
template <typename Dtype>
__global__ void PixelwiseCorrelationForward(int num, int channel, int width, 
    int height, int kernel_size, int half_kernel, int chs_idx, int total_chs,
	const Dtype* bottom_data, Dtype* top_data) 
{
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  //pixel location in the new image
  int n_idx = thread_idx / (channel*height*width);
  int n_res = thread_idx % (channel*height*width);
  int c_idx = n_res / (height*width);
  int c_res = n_res % (height*width);
  int h_idx = c_res / width;
  int w_idx = c_res % width;
  if (n_idx >= num || c_idx >= channel || h_idx >= height || w_idx >= width) 
    return;
  
  int channel_idx = 0;
  for(int i = -1*half_kernel; i <= half_kernel; i++){
    for(int j = -1*half_kernel; j <= half_kernel; j++){
	  int h = h_idx + i;
	  int w = w_idx + j;
	  if(h < 0) continue; if(h >= height) continue;
	  if(w < 0) continue; if(w >= width)  continue;
	  int overlap_count = 0;
	  Dtype corr_ij(0.0);
	  for(int q = -1*half_kernel; q <= half_kernel; q++){
	    for(int m = -1*half_kernel; m <= half_kernel; m++){
		  int q_i = q - i;
		  int m_j = m - j;
		  if(abs(q_i)<=half_kernel && abs(m_j)<=half_kernel){
		    int hh = h_idx + q_i;
			int ww = w_idx + m_j;
			if(hh < 0) continue; if(hh >= height) continue;
	        if(ww < 0) continue; if(ww >= width)  continue;
		    overlap_count++;
			corr_ij += bottom_data[n_idx*channel*height*width+c_idx*height*width+h*width+w]*
			    bottom_data[n_idx*channel*height*width+c_idx*height*width+hh*width+ww];
		  }
		}
	  }
	  if(overlap_count!=0){
	    corr_ij /= overlap_count;
	  }
	  top_data[n_idx*total_chs*height*width+((chs_idx+channel_idx)*channel+c_idx)*height*width+h_idx*width+w_idx] = corr_ij;
	  channel_idx++;
	}
  }
}


template <typename Dtype>
void PixelwiseCorrelationLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  top[0]->scale_data(0);
  
  int num     = bottom[0]->num();
  int channel = bottom[0]->channels();
  int height  = bottom[0]->height();
  int width   = bottom[0]->width();
  int count   = bottom[0]->count();
  
  int chs_idx = 0;
  for(int i = 0; i < kernel_sizes_.size(); i++){
    int kernel_size = kernel_sizes_[i];
	int half_kernel = kernel_size/2;
	PixelwiseCorrelationForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        num, channel, width, height, kernel_size, half_kernel,
	    chs_idx, total_channels_,
	    bottom[0]->gpu_data(), 
		top[0]->mutable_gpu_data());
	chs_idx += kernel_size*kernel_size;
    CUDA_POST_KERNEL_CHECK;
  }
}

// one thread for every pixel
template <typename Dtype>
__global__ void PixelwiseCorrelationBackward(int num, int channel, int width, 
    int height, int kernel_size, int half_kernel, int chs_idx, int total_chs,
	const Dtype* bottom_data, const Dtype* top_data, 
	Dtype* bottom_diff, const Dtype* top_diff) 
{
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  //pixel location in the new image
  int n_idx = thread_idx / (channel*height*width);
  int n_res = thread_idx % (channel*height*width);
  int c_idx = n_res / (height*width);
  int c_res = n_res % (height*width);
  int h_idx = c_res / width;
  int w_idx = c_res % width;
  if (n_idx >= num || c_idx >= channel || h_idx >= height || w_idx >= width) 
    return;
  
  int channel_idx = 0;
  for(int i = -1*half_kernel; i <= half_kernel; i++){
    for(int j = -1*half_kernel; j <= half_kernel; j++){
	  int h = h_idx + i;
	  int w = w_idx + j;
	  if(h < 0) continue; if(h >= height) continue;
	  if(w < 0) continue; if(w >= width)  continue;
	  int overlap_count = 0;
	  Dtype corr_ij(0.0);
	  for(int q = -1*half_kernel; q <= half_kernel; q++){
	    for(int m = -1*half_kernel; m <= half_kernel; m++){
		  int q_i = q - i;
		  int m_j = m - j;
		  if(abs(q_i)<=half_kernel && abs(m_j)<=half_kernel){
		    int hh = h_idx + q_i;
			int ww = w_idx + m_j;
			if(hh < 0) continue; if(hh >= height) continue;
	        if(ww < 0) continue; if(ww >= width)  continue;
		    overlap_count++;
			corr_ij += bottom_data[n_idx*channel*height*width+c_idx*height*width+h*width+w]*
			    bottom_data[n_idx*channel*height*width+c_idx*height*width+hh*width+ww];
		  }
		}
	  }
	  if(overlap_count!=0){
	    corr_ij /= overlap_count;
	  }
	  Dtype ct_top_diff = 
	      top_diff[n_idx*total_chs*height*width+((chs_idx+channel_idx)*channel+c_idx)*height*width+h_idx*width+w_idx]/
		  overlap_count;
	  for(int q = -1*half_kernel; q <= half_kernel; q++){
	    for(int m = -1*half_kernel; m <= half_kernel; m++){
		  int q_i = q - i;
		  int m_j = m - j;
		  if(abs(q_i)<=half_kernel && abs(m_j)<=half_kernel){
		    int hh = h_idx + q_i;
			int ww = w_idx + m_j;
			if(hh < 0) continue; if(hh >= height) continue;
	        if(ww < 0) continue; if(ww >= width)  continue;
			caffe_gpu_atomic_add(Dtype(ct_top_diff*bottom_data[n_idx*channel*height*width+c_idx*height*width+hh*width+ww]), 
	            bottom_diff+n_idx*channel*height*width+c_idx*height*width+h*width+w);
			caffe_gpu_atomic_add(Dtype(ct_top_diff*bottom_data[n_idx*channel*height*width+c_idx*height*width+h*width+w]), 
	            bottom_diff+n_idx*channel*height*width+c_idx*height*width+hh*width+ww);
		  }
		}
	  }	  
	  channel_idx++;
	}
  }
}

template <typename Dtype>
void PixelwiseCorrelationLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  bottom[0]->scale_diff(0);
  
  int num     = bottom[0]->num();
  int channel = bottom[0]->channels();
  int height  = bottom[0]->height();
  int width   = bottom[0]->width();
  int count   = bottom[0]->count();
  
  int chs_idx = 0;
  for(int i = 0; i < kernel_sizes_.size(); i++){
    int kernel_size = kernel_sizes_[i];
	int half_kernel = kernel_size/2;
	PixelwiseCorrelationBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        num, channel, width, height, kernel_size, half_kernel,
	    chs_idx, total_channels_,
	    bottom[0]->gpu_data(), 
		top[0]->gpu_data(),
	    bottom[0]->mutable_gpu_diff(), 
		top[0]->gpu_diff());
	chs_idx += kernel_size*kernel_size;
    CUDA_POST_KERNEL_CHECK;
  }
}


INSTANTIATE_LAYER_GPU_FUNCS(PixelwiseCorrelationLayer);


}  // namespace caffe