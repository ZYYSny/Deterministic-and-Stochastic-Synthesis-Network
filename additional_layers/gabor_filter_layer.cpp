#ifdef USE_CUDNN
#include <algorithm>
#include <vector>
#include <fstream>
#include <sstream>

#include "caffe/layers/gabor_filter_layer.hpp"

#define PI 3.1415926

namespace caffe {

// Set to three for the benefit of the backward pass, which
// can use separate streams for calculating the gradient w.r.t.
// bias, filter weights, and bottom data for each group independently
#define CUDNN_STREAMS_PER_GROUP 3
/**
* TODO(dox) explain cuDNN interface
*/
template <typename Dtype>
void GaborFilterLayer<Dtype>::LayerSetUp(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	ConvolutionLayer<Dtype>::LayerSetUp(bottom, top);
	// Initialize CUDA streams and cuDNN.
	stream_ = new cudaStream_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
	handle_ = new cudnnHandle_t[this->group_ * CUDNN_STREAMS_PER_GROUP];
	// Initialize algorithm arrays
	fwd_algo_ = new cudnnConvolutionFwdAlgo_t[bottom.size()];
	bwd_filter_algo_ = new cudnnConvolutionBwdFilterAlgo_t[bottom.size()];
	bwd_data_algo_ = new cudnnConvolutionBwdDataAlgo_t[bottom.size()];
	// initialize size arrays
	workspace_fwd_sizes_ = new size_t[bottom.size()];
	workspace_bwd_filter_sizes_ = new size_t[bottom.size()];
	workspace_bwd_data_sizes_ = new size_t[bottom.size()];
	// workspace data
	workspaceSizeInBytes = 0;
	workspaceData = NULL;
	workspace = new void*[this->group_ * CUDNN_STREAMS_PER_GROUP];
	for (size_t i = 0; i < bottom.size(); ++i) {
		// initialize all to default algorithms
		fwd_algo_[i] = (cudnnConvolutionFwdAlgo_t)0;
		bwd_filter_algo_[i] = (cudnnConvolutionBwdFilterAlgo_t)0;
		bwd_data_algo_[i] = (cudnnConvolutionBwdDataAlgo_t)0;
		// default algorithms don't require workspace
		workspace_fwd_sizes_[i] = 0;
		workspace_bwd_data_sizes_[i] = 0;
		workspace_bwd_filter_sizes_[i] = 0;
	}
	for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
		CUDA_CHECK(cudaStreamCreate(&stream_[g]));
		CUDNN_CHECK(cudnnCreate(&handle_[g]));
		CUDNN_CHECK(cudnnSetStream(handle_[g], stream_[g]));
		workspace[g] = NULL;
	}
	// Set the indexing parameters.
	bias_offset_ = (this->num_output_ / this->group_);
	// Create filter descriptor.
	const int* kernel_shape_data = this->kernel_shape_.cpu_data();
	const int kernel_h = kernel_shape_data[0];
	const int kernel_w = kernel_shape_data[1];
	cudnn::createFilterDesc<Dtype>(&filter_desc_,
    	this->num_output_ / this->group_, this->channels_ / this->group_,
		kernel_h, kernel_w);
	// Create tensor descriptor(s) for data and corresponding convolution(s).
	for (int i = 0; i < bottom.size(); i++) {
		cudnnTensorDescriptor_t bottom_desc;
		cudnn::createTensor4dDesc<Dtype>(&bottom_desc);
		bottom_descs_.push_back(bottom_desc);
		cudnnTensorDescriptor_t top_desc;
		cudnn::createTensor4dDesc<Dtype>(&top_desc);
		top_descs_.push_back(top_desc);
		cudnnConvolutionDescriptor_t conv_desc;
		cudnn::createConvolutionDesc<Dtype>(&conv_desc);
		conv_descs_.push_back(conv_desc);
	}
	// Tensor descriptor for bias.
	if (this->bias_term_) {
		cudnn::createTensor4dDesc<Dtype>(&bias_desc_);
	}
	handles_setup_ = true;

	//gabor filter parameters
	GaborParameter gabor_filter = this->layer_param_.gabor_param();
	space_ = gabor_filter.space();
	upper_frequency_ = gabor_filter.upper_frequency();
	lower_frequency_ = gabor_filter.lower_frequency();
	scale_num_ = gabor_filter.scale_num();
	orientation_ = gabor_filter.orientation();
	remove_dc_ = gabor_filter.remove_dc();
	show_filters_ = gabor_filter.show_filters();
	x_scale_ = gabor_filter.x_scale();
	y_scale_ = gabor_filter.y_scale();
	CHECK_GT(x_scale_, 0);
	CHECK_GT(y_scale_, 0);
	//
	if(gabor_filter.has_real_part_path()){
		real_part_path_ = gabor_filter.real_part_path();
	}else{
		real_part_path_ = "gabor_filter_real.jpg";
	}
	//
	if(gabor_filter.has_imag_part_path()){
		imag_part_path_ = gabor_filter.imag_part_path();
	}else{
		imag_part_path_ = "gabor_filter_imag.jpg";
	}
	//setup gabor filters
	shared_ptr<Blob<Dtype> > weights = this->blobs_[0];
	CHECK_EQ(this->blobs_.size(),1);
	CHECK_EQ(2 * scale_num_ * orientation_, weights->num());
	CHECK_EQ(1, weights->channels());
	CHECK_EQ(space_, weights->width());
	CHECK_EQ(space_, weights->height());
	CHECK_EQ(1, space_%2);
	//gabor filter visualization
	int interval = 20;
	double scale = 1;
	int scaled_rows = int(scale*space_);
    int scaled_cols = int(scale*space_);
	cv::Mat RealGabor = cv::Mat::ones(scale_num_*(scaled_rows+interval)+interval, 
        orientation_*(scaled_cols+interval)+interval, CV_8UC3);
  cv::Mat ImagGabor = cv::Mat::ones(scale_num_*(scaled_rows+interval)+interval, 
        orientation_*(scaled_cols+interval)+interval, CV_8UC3);
  RealGabor = RealGabor * 255;
  ImagGabor = ImagGabor * 255;
	//
  std::ofstream gabor("gabor.txt");
	for(int k = 0; k < weights->num()/2; k++){
		cv::Mat gabor_filter_real = cv::Mat::zeros(space_, space_, CV_32FC1);
		cv::Mat gabor_filter_imag = cv::Mat::zeros(space_, space_, CV_32FC1);
		
		int s = k/orientation_ + 1;
		int n = k%orientation_ + 1;
		
		double base, a, u0, z, Uvar, Vvar, Xvar, Yvar, X, Y, G, t1, t2, m, t;
        int x, y, side;
        base = upper_frequency_/lower_frequency_;
        a = pow(base, 1.0/(double)(scale_num_-1));
        u0 = upper_frequency_/pow(a, (double) scale_num_ - s);
        Uvar = (a-1.0)*u0/((a+1.0)*sqrt(2.0*log(2.0)));
        z = -2.0*log(2.0)*(Uvar*Uvar)/u0;
        Vvar = tan(PI/(2*orientation_))*(u0+z)/sqrt(2.0*log(2.0)-z*z/(Uvar*Uvar));
        Xvar = 1.0/(2.0*PI*Uvar)*x_scale_;
        Yvar = 1.0/(2.0*PI*Vvar)*y_scale_;
        t1 = cos(PI/orientation_*(n-1.0));
        t2 = sin(PI/orientation_*(n-1.0));
        side = (int) (space_-1)/2;
        
        for (x=0;x<2*side+1;x++) {
	        for (y=0;y<2*side+1;y++) {
		        X = (double) (x-side)*t1+ (double) (y-side)*t2;
		        Y = (double) -(x-side)*t2+ (double) (y-side)*t1;
		        G = 1.0/(2.0*PI*Xvar*Yvar)*pow(a, (double) scale_num_-s)*exp(-0.5*((X*X)/(Xvar*Xvar)+(Y*Y)/(Yvar*Yvar)));
		        gabor_filter_real.at<float>(y,x) = G*cos(2.0*PI*u0*X);
	            gabor_filter_imag.at<float>(y,x) = G*sin(2.0*PI*u0*X);
	        }
        }
        /* if flag = 1, then remove the DC from the filter */
        if (remove_dc_ == 1) {
	        m = 0;
			t = 0;
	        for (x=0;x<2*side+1;x++)
		        for (y=0;y<2*side+1;y++){
					m += gabor_filter_real.at<float>(x,y);
					t += gabor_filter_imag.at<float>(x,y);
				}
			        
	        m /= pow((double) 2.0*side+1, 2.0);
			t /= pow((double) 2.0*side+1, 2.0);
	        for (x=0;x<2*side+1;x++)
		        for (y=0;y<2*side+1;y++){
					gabor_filter_real.at<float>(x,y) -= m;
					gabor_filter_imag.at<float>(x,y) -= t;
				}
            //					
        }
		//Caculate the output range of Gabor Filter
    gabor<<"The "<<k<<"th gabor filter"<<std::endl;
    gabor<<"REAL"<<std::endl;
		Dtype minOutput(0.0), maxOutput(0.0);
		for (x=0;x<2*side+1;x++){
		        for (y=0;y<2*side+1;y++){
          //LOG(INFO)<<gabor_filter_real.at<float>(x,y)<<" "<<gabor_filter_imag.at<float>(x,y);
					if(gabor_filter_real.at<float>(x,y)>0){
						minOutput -= 255*gabor_filter_real.at<float>(x,y);
						maxOutput += 255*gabor_filter_real.at<float>(x,y);
					}else{
						minOutput += 255*gabor_filter_real.at<float>(x,y);
						maxOutput -= 255*gabor_filter_real.at<float>(x,y);
					}
           gabor<<gabor_filter_real.at<float>(x,y)<<"\t";
				}
        gabor<<std::endl;
    }
		LOG(INFO)<<"The "<< k <<"th Gabor Filter REAL part output range: from "<< minOutput<<" to "<<maxOutput;
		//
		minOutput=Dtype(0.0);
		maxOutput=Dtype(0.0);
    gabor<<"IMAG"<<std::endl;
		for (x=0;x<2*side+1;x++){
		        for (y=0;y<2*side+1;y++){
					if(gabor_filter_imag.at<float>(x,y)>0){
						minOutput -= 255*gabor_filter_imag.at<float>(x,y);
						maxOutput += 255*gabor_filter_imag.at<float>(x,y);
					}else{
						minOutput += 255*gabor_filter_imag.at<float>(x,y);
						maxOutput -= 255*gabor_filter_imag.at<float>(x,y);
					}
           gabor<<gabor_filter_imag.at<float>(x,y)<<"\t";
				}
        gabor<<std::endl;
    }
		LOG(INFO)<<"The "<< k <<"th Gabor Filter IMAG part output range: from "<< minOutput<<" to "<<maxOutput;
		//
		Dtype * real_weights_data = weights->mutable_cpu_data()+ k * space_ * space_ * sizeof(Dtype);
        Dtype * imag_weights_data = weights->mutable_cpu_data()+(k + weights->num()/2) * space_ * space_ * sizeof(Dtype);
        //visualize the gabor filters
        float realMaxEle = -1*(INT_MAX);
        float realMinEle = INT_MAX;
		float imagMaxEle = -1*(INT_MAX);
        float imagMinEle = INT_MAX;
        float *real_img_data = (float*)(gabor_filter_real.data);
		float *imag_img_data = (float*)(gabor_filter_imag.data);
		int count = gabor_filter_real.rows * gabor_filter_real.cols;
        for(int p =0;p<count;p++){
            realMaxEle = std::max(realMaxEle, real_img_data[p]);
            realMinEle = std::min(realMinEle, real_img_data[p]);
    		imagMaxEle = std::max(imagMaxEle, imag_img_data[p]);
            imagMinEle = std::min(imagMinEle, imag_img_data[p]);
	    	weights->mutable_cpu_data()[k * space_ * space_ + p] = Dtype(real_img_data[p]);
			weights->mutable_cpu_data()[(k + weights->num()/2) * space_ * space_ + p] = Dtype(imag_img_data[p]);
        }
		cv::Mat RealFilter = cv::Mat::zeros(space_, space_, CV_8UC1);
		cv::Mat ImagFilter = cv::Mat::zeros(space_, space_, CV_8UC1);
		uchar *real_filter_data = (uchar*)(RealFilter.data);
		uchar *imag_filter_data = (uchar*)(ImagFilter.data);
		//
		const Dtype * real_weights_checked_data = weights->cpu_data()+ k * space_ * space_ ;
		const Dtype * imag_weights_checked_data = weights->cpu_data()+ (k + weights->num()/2) * space_ * space_ ;
		for(m = 0; m < count; m++){
            real_filter_data[(int)m] = (int)(255 * (real_weights_checked_data[(int)m]-realMinEle)/(realMaxEle-realMinEle));
			imag_filter_data[(int)m] = (int)(255 * (imag_weights_checked_data[(int)m]-imagMinEle)/(imagMaxEle-imagMinEle));
		}
		//LOG(INFO)<<s<<" "<<n<<" "<<base<<" "<<a<<" "<<u0<<" "<<Uvar << " " << z << " " << Vvar << " " << Xvar << " " << Yvar << " " << t1 << " " << t2 << " " << side;
		cv::applyColorMap(RealFilter, RealFilter, cv::COLORMAP_JET);
		cv::applyColorMap(ImagFilter, ImagFilter, cv::COLORMAP_JET);
		cv::resize(RealFilter, RealFilter, cv::Size(scaled_cols, scaled_rows));
		cv::resize(ImagFilter, ImagFilter, cv::Size(scaled_cols, scaled_rows));
		cv::Rect real_roi = cv::Rect((n-1)*(scaled_cols+interval)+interval, 
		    (s-1)*(scaled_rows+interval)+interval,
            RealFilter.rows,RealFilter.cols);
        cv::resize(RealFilter, RealGabor(real_roi), RealFilter.size());
		cv::Rect imag_roi = cv::Rect((n-1)*(scaled_cols+interval)+interval, 
			(s-1)*(scaled_rows+interval)+interval,
            ImagFilter.rows,ImagFilter.cols);
        cv::resize(ImagFilter, ImagGabor(imag_roi), ImagFilter.size());
	    //
	  }
    if(show_filters_){
        cv::imshow("Real Part of Gabor Filter!", RealGabor);
	    cv::imshow("Imag Part of Gabor Filter!", ImagGabor);
	    cv::waitKey(0);
    }
	cv::imwrite(real_part_path_, RealGabor);
    cv::imwrite(imag_part_path_, ImagGabor);
}

template <typename Dtype>
void GaborFilterLayer<Dtype>::Reshape(
	const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	ConvolutionLayer<Dtype>::Reshape(bottom, top);
	CHECK_EQ(2, this->num_spatial_axes_)
		<< "CuDNNConvolution input must have 2 spatial axes "
		<< "(e.g., height and width). "
		<< "Use 'engine: CAFFE' for general ND convolution.";
	bottom_offset_ = this->bottom_dim_ / this->group_;
	top_offset_ = this->top_dim_ / this->group_;
	const int height = bottom[0]->shape(this->channel_axis_ + 1);
	const int width = bottom[0]->shape(this->channel_axis_ + 2);
	const int height_out = top[0]->shape(this->channel_axis_ + 1);
	const int width_out = top[0]->shape(this->channel_axis_ + 2);
	const int* pad_data = this->pad_.cpu_data();
	const int pad_h = pad_data[0];
	const int pad_w = pad_data[1];
	const int* stride_data = this->stride_.cpu_data();
	const int stride_h = stride_data[0];
	const int stride_w = stride_data[1];
	// Specify workspace limit for kernels directly until we have a
	// planning strategy and a rewrite of Caffe's GPU memory mangagement
	size_t workspace_limit_bytes = 8 * 1024 * 1024;
	for (int i = 0; i < bottom.size(); i++) {
		cudnn::setTensor4dDesc<Dtype>(&bottom_descs_[i],
			this->num_,
			this->channels_ / this->group_, height, width,
			this->channels_ * height * width,
			height * width, width, 1);
		cudnn::setTensor4dDesc<Dtype>(&top_descs_[i],
			this->num_,
			this->num_output_ / this->group_, height_out, width_out,
			this->num_output_ * this->out_spatial_dim_,
			this->out_spatial_dim_, width_out, 1);
		cudnn::setConvolutionDesc<Dtype>(&conv_descs_[i], bottom_descs_[i],
			filter_desc_, pad_h, pad_w,
			stride_h, stride_w);
			// choose forward and backward algorithms + workspace(s)
		CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(handle_[0],
			bottom_descs_[i],
			filter_desc_,
			conv_descs_[i],
			top_descs_[i],
			CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
			workspace_limit_bytes,
			&fwd_algo_[i]));
		CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_[0],
			bottom_descs_[i],
			filter_desc_,
			conv_descs_[i],
			top_descs_[i],
			fwd_algo_[i],
			&(workspace_fwd_sizes_[i])));
		// choose backward algorithm for filter
		CUDNN_CHECK(cudnnGetConvolutionBackwardFilterAlgorithm(handle_[0],
			bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
			CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
			workspace_limit_bytes, &bwd_filter_algo_[i]));
		// get workspace for backwards filter algorithm
		CUDNN_CHECK(cudnnGetConvolutionBackwardFilterWorkspaceSize(handle_[0],
			bottom_descs_[i], top_descs_[i], conv_descs_[i], filter_desc_,
			bwd_filter_algo_[i], &workspace_bwd_filter_sizes_[i]));
		// choose backward algo for data
		CUDNN_CHECK(cudnnGetConvolutionBackwardDataAlgorithm(handle_[0],
			filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
			CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
			workspace_limit_bytes, &bwd_data_algo_[i]));
		// get workspace size
		CUDNN_CHECK(cudnnGetConvolutionBackwardDataWorkspaceSize(handle_[0],
			filter_desc_, top_descs_[i], conv_descs_[i], bottom_descs_[i],
			bwd_data_algo_[i], &workspace_bwd_data_sizes_[i]));
	}

	// reduce over all workspace sizes to get a maximum to allocate / reallocate
	size_t total_workspace_fwd = 0;
	size_t total_workspace_bwd_data = 0;
	size_t total_workspace_bwd_filter = 0;
	for (size_t i = 0; i < bottom.size(); i++) {
		total_workspace_fwd = std::max(total_workspace_fwd,
    		workspace_fwd_sizes_[i]);
		total_workspace_bwd_data = std::max(total_workspace_bwd_data,
			workspace_bwd_data_sizes_[i]);
		total_workspace_bwd_filter = std::max(total_workspace_bwd_filter,
			workspace_bwd_filter_sizes_[i]);
	}
	// get max over all operations
	size_t max_workspace = std::max(total_workspace_fwd,
		total_workspace_bwd_data);
	max_workspace = std::max(max_workspace, total_workspace_bwd_filter);
	// ensure all groups have enough workspace
	size_t total_max_workspace = max_workspace *
		(this->group_ * CUDNN_STREAMS_PER_GROUP);
		// this is the total amount of storage needed over all groups + streams
	if (total_max_workspace > workspaceSizeInBytes) {
		DLOG(INFO) << "Reallocating workspace storage: " << total_max_workspace;
		workspaceSizeInBytes = total_max_workspace;
			// free the existing workspace and allocate a new (larger) one
		cudaFree(this->workspaceData);
		cudaError_t err = cudaMalloc(&(this->workspaceData), workspaceSizeInBytes);
		if (err != cudaSuccess) {
			// force zero memory path
			for (int i = 0; i < bottom.size(); i++) {
				workspace_fwd_sizes_[i] = 0;
				workspace_bwd_filter_sizes_[i] = 0;
				workspace_bwd_data_sizes_[i] = 0;
				fwd_algo_[i] = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
				bwd_filter_algo_[i] = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
				bwd_data_algo_[i] = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
			}
			// NULL out all workspace pointers
			for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
				workspace[g] = NULL;
			}
			// NULL out underlying data
			workspaceData = NULL;
			workspaceSizeInBytes = 0;
		}

		// if we succeed in the allocation, set pointer aliases for workspaces
		for (int g = 0; g < (this->group_ * CUDNN_STREAMS_PER_GROUP); g++) {
			workspace[g] = reinterpret_cast<char *>(workspaceData) + g*max_workspace;
		}
	}

	// Tensor descriptor for bias.
	if (this->bias_term_) {
		cudnn::setTensor4dDesc<Dtype>(&bias_desc_,
			1, this->num_output_ / this->group_, 1, 1);
	}
}

template <typename Dtype>
GaborFilterLayer<Dtype>::~GaborFilterLayer() {
	// Check that handles have been setup before destroying.
	if (!handles_setup_) { return; }
	for (int i = 0; i < bottom_descs_.size(); i++) {
		cudnnDestroyTensorDescriptor(bottom_descs_[i]);
		cudnnDestroyTensorDescriptor(top_descs_[i]);
		cudnnDestroyConvolutionDescriptor(conv_descs_[i]);
	}
	if (this->bias_term_) {
		cudnnDestroyTensorDescriptor(bias_desc_);
	}
	cudnnDestroyFilterDescriptor(filter_desc_);
	for (int g = 0; g < this->group_ * CUDNN_STREAMS_PER_GROUP; g++) {
		cudaStreamDestroy(stream_[g]);
		cudnnDestroy(handle_[g]);
	}

	cudaFree(workspaceData);
	delete[] stream_;
	delete[] handle_;
	delete[] fwd_algo_;
	delete[] bwd_filter_algo_;
	delete[] bwd_data_algo_;
	delete[] workspace_fwd_sizes_;
	delete[] workspace_bwd_data_sizes_;
	delete[] workspace_bwd_filter_sizes_;
}

INSTANTIATE_CLASS(GaborFilterLayer);
REGISTER_LAYER_CLASS(GaborFilter);
}   // namespace caffe
#endif
