#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pixelwise_grammatrix_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/gpu_util.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_functions.h>

namespace caffe {

// one thread for every pixel
template <typename Dtype>
__global__ void PixelwiseGrammatrixForward(int num, int channel,
    int width, int height, int kernel_size, int half_kernel, 
	const Dtype* source_data, 
	const Dtype* target_data,
	Dtype* diff, Dtype* loss) 
{
  int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  //pixel location in the new image
  int n_idx = thread_idx / (channel*channel*height*width);
  int n_res = thread_idx % (channel*channel*height*width);
  int c_idx = n_res / (channel*height*width);
  int c_res = n_res % (channel*height*width);
  int c_jdx = c_res / (height*width);
  int j_res = c_res % (height*width);
  int h_idx = j_res / width;
  int w_idx = j_res % width;
  if (n_idx >= num || c_idx >= channel || c_jdx >= channel || h_idx >= height || w_idx >= width) 
    return;
  
  Dtype source_gram_value(0.0);
  Dtype target_gram_value(0.0);
  for(int i = -1*half_kernel; i <= half_kernel; i++){
    for(int j = -1*half_kernel; j <= half_kernel; j++){
	  int h = h_idx + i;
	  int w = w_idx + j;
	  if(h < 0) h = 0; if(h >= height) h = height - 1;
	  if(w < 0) w = 0; if(w >= width)  w = width  - 1;
	  source_gram_value += source_data[n_idx*channel*height*width+c_idx*height*width+h*width+w]*
	      source_data[n_idx*channel*height*width+c_jdx*height*width+h*width+w];
	  target_gram_value += target_data[n_idx*channel*height*width+c_idx*height*width+h*width+w]*
	      target_data[n_idx*channel*height*width+c_jdx*height*width+h*width+w];
	}
  }
  source_gram_value /= (kernel_size*kernel_size);
  target_gram_value /= (kernel_size*kernel_size);
  //calculate loss
  Dtype current_loss(0.0);
  current_loss = (source_gram_value-target_gram_value)*(source_gram_value-target_gram_value)/
      (4*channel*channel*height*width);
  caffe_gpu_atomic_add(current_loss, 
      loss+n_idx*channel*height*width+c_idx*height*width+h_idx*width+w_idx);
  //calculate gradient
  Dtype source_gram_diff = 
      (source_gram_value-target_gram_value) / (2*channel*channel*height*width);
  source_gram_diff /= (kernel_size*kernel_size);
  for(int i = -1*half_kernel; i <= half_kernel; i++){
    for(int j = -1*half_kernel; j <= half_kernel; j++){
	  int h = h_idx + i;
	  int w = w_idx + j;
	  if(h < 0) h = 0; if(h >= height) h = height - 1;
	  if(w < 0) w = 0; if(w >= width)  w = width  - 1;

	  caffe_gpu_atomic_add(
	      Dtype(source_gram_diff*source_data[n_idx*channel*height*width+c_jdx*height*width+h*width+w]), 
          diff+n_idx*channel*height*width+c_idx*height*width+h*width+w);
	  caffe_gpu_atomic_add(
	      Dtype(source_gram_diff*source_data[n_idx*channel*height*width+c_idx*height*width+h*width+w]), 
          diff+n_idx*channel*height*width+c_jdx*height*width+h*width+w);
	}
  }  
}


template <typename Dtype>
void PixelwiseGrammatrixLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* source_data = bottom[0]->gpu_data();
  const Dtype* target_data = bottom[1]->gpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  
  //set data
  diff_.scale_data(0);
  loss_.scale_data(0);

  int num     = bottom[0]->num();
  int channel = bottom[0]->channels();
  int height  = bottom[0]->height();
  int width   = bottom[0]->width();
  int count   = bottom[0]->count()*channel;
  
  for(int i = 0; i < kernel_sizes_.size(); i++){
    int kernel_size = kernel_sizes_[i];
	int half_kernel = kernel_size/2;
	PixelwiseGrammatrixForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        num, channel, width, height, kernel_size, half_kernel,
	    source_data, target_data,
	    diff_.mutable_gpu_data(), 
		loss_.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
  }
  top_data[0] = loss_.asum_data();
}

template <typename Dtype>
void PixelwiseGrammatrixLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  Dtype loss_weight = top[0]->cpu_diff()[0];
  caffe_gpu_scale(diff_.count(), loss_weight, 
      diff_.gpu_data(), bottom[0]->mutable_gpu_diff());
}


INSTANTIATE_LAYER_GPU_FUNCS(PixelwiseGrammatrixLayer);


}  // namespace caffe