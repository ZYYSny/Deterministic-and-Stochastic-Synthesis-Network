#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pixel_shuffle_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

// one thread for every pixel
template <typename Dtype>
__global__ void PixelShuffleForward(int num, int channel, int width, 
    int height, int scale, int scale_sqrt, const Dtype* bottom_data, Dtype* top_data) 
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
  //pixel location in the ori image
  int cc_idx = c_idx / scale;
  int cc_res = c_idx % scale;
  int hs_idx = cc_res / scale_sqrt;
  int ws_idx = cc_res % scale_sqrt;
  int hh_idx = scale_sqrt * h_idx + hs_idx;
  int ww_idx = scale_sqrt * w_idx + ws_idx;
  top_data[n_idx*(channel*height*width)+cc_idx*scale*height*width+hh_idx*scale_sqrt*width+ww_idx] = 
      bottom_data[thread_idx];
}


template <typename Dtype>
void PixelShuffleLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  int count = top[0]->count();
  PixelShuffleForward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    bottom[0]->num(), bottom[0]->channels(), bottom[0]->width(), bottom[0]->height(),
	scale_,scale_sqrt_,bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK;
  const Dtype* bottom_cpu_data = bottom[0]->cpu_data();
  bottom[0]->release_gpu_data();
  const Dtype* top_cpu_data = top[0]->cpu_data();
  top[0]->release_gpu_data();
}


// one thread for every pixel
template <typename Dtype>
__global__ void PixelShuffleBackward(int num, int channel, int width, 
    int height, int scale, int scale_sqrt, Dtype* bottom_diff, const Dtype* top_diff) 
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
  //pixel location in the ori image
  int cc_idx = c_idx / scale;
  int cc_res = c_idx % scale;
  int hs_idx = cc_res / scale_sqrt;
  int ws_idx = cc_res % scale_sqrt;
  int hh_idx = scale_sqrt * h_idx + hs_idx;
  int ww_idx = scale_sqrt * w_idx + ws_idx;
  bottom_diff[thread_idx] = 
     top_diff[n_idx*(channel*height*width)+cc_idx*scale*height*width+hh_idx*scale_sqrt*width+ww_idx];
}


template <typename Dtype>
void PixelShuffleLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->gpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  const int count = top[0]->count();
  PixelShuffleBackward<<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
    bottom[0]->num(), bottom[0]->channels(), bottom[0]->width(), bottom[0]->height(),
	scale_,scale_sqrt_,bottom_diff, top_diff);
  CUDA_POST_KERNEL_CHECK;
}


INSTANTIATE_LAYER_GPU_FUNCS(PixelShuffleLayer);


}  // namespace caffe