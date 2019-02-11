#ifndef CAFFE_GABOR_FILTER_LAYER_HPP_
#define CAFFE_GABOR_FILTER_LAYER_HPP_

#include <vector>
#include <opencv2/opencv.hpp>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

#ifdef USE_CUDNN
/*
* @brief cuDNN implementation of ConvolutionLayer.
*        Fallback to ConvolutionLayer for CPU mode.
*
* cuDNN accelerates convolution through forward kernels for filtering and bias
* plus backward kernels for the gradient w.r.t. the filters, biases, and
* inputs. Caffe + cuDNN further speeds up the computation through forward
* parallelism across groups and backward parallelism across gradients.
*
* The CUDNN engine does not have memory overhead for matrix buffers. For many
* input and filter regimes the CUDNN engine is faster than the CAFFE engine,
* but for fully-convolutional models and large inputs the CAFFE engine can be
* faster as long as it fits in memory.
*/
template <typename Dtype>
class GaborFilterLayer : public ConvolutionLayer<Dtype> {
 public:
	explicit GaborFilterLayer(const LayerParameter& param)
	: ConvolutionLayer<Dtype>(param), handles_setup_(false) {}
	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual ~GaborFilterLayer();
	virtual inline const char* type() const { return "GaborFilter"; }
 protected:
	virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
		const vector<Blob<Dtype>*>& top);
	virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
	bool handles_setup_;
	cudnnHandle_t* handle_;
	cudaStream_t*  stream_;
	// algorithms for forward and backwards convolutions
	cudnnConvolutionFwdAlgo_t *fwd_algo_;
	cudnnConvolutionBwdFilterAlgo_t *bwd_filter_algo_;
	cudnnConvolutionBwdDataAlgo_t *bwd_data_algo_;
	vector<cudnnTensorDescriptor_t> bottom_descs_, top_descs_;
	cudnnTensorDescriptor_t    bias_desc_;
	cudnnFilterDescriptor_t      filter_desc_;
	vector<cudnnConvolutionDescriptor_t> conv_descs_;
	int bottom_offset_, top_offset_, bias_offset_;
	size_t *workspace_fwd_sizes_;
	size_t *workspace_bwd_data_sizes_;
	size_t *workspace_bwd_filter_sizes_;
	size_t workspaceSizeInBytes;  // size of underlying storage
	void *workspaceData;  // underlying storage
	void **workspace;  // aliases into workspaceData
	//Gabor Filer Parameters
	int space_;//kernel size
	double upper_frequency_;
	double lower_frequency_;
	int scale_num_;
	int orientation_;
	bool remove_dc_;
	bool show_filters_;
	string real_part_path_;
	string imag_part_path_;
    cv::Mat real_gabor_;
    cv::Mat imag_gabor_;
	int x_scale_;
	int y_scale_;
};
#endif
}  // namespace caffe

#endif  // CAFFE_GABOR_FILTER_LAYER_HPP_
