#include <vector>
#include <math.h>
#include <sstream>
#include <string>

#include "caffe/layers/visualize_layer.hpp"
#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void VisualizeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  VisualizeParameter visualize_param = this->layer_param_.visualize_param();
  width_ = visualize_param.width();
  height_ = visualize_param.height();
  gray_ = visualize_param.gray();
  scale_ = visualize_param.scale();
  display_interval_ = visualize_param.display_interval();
  if(!visualize_param.has_root_folder())LOG(FATAL)<<"ROOT_FOLDER should be assigned!";
  root_folder_ = visualize_param.root_folder();
  CHECK_EQ(bottom.size(), 2);
  CHECK_EQ(bottom[0]->count(), bottom[1]->count());
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  //CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  //CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  //CHECK_EQ(bottom[0]->width(), width_);
  //CHECK_EQ(bottom[0]->height(), height_);
  if(!gray_){
    CHECK_EQ(bottom[0]->channels(), 3);
  }
  //
  transform_param_ = this->layer_param_.transform_param();
  if (transform_param_.has_mean_file()) {
    CHECK_EQ(transform_param_.mean_value_size(), 0) <<
      "Cannot specify mean_file and mean_value at the same time";
    const string& mean_file = transform_param_.mean_file();
    if (Caffe::root_solver()) {
      LOG(INFO) << "Loading mean file from: " << mean_file;
    }
    BlobProto blob_proto;
    ReadProtoFromBinaryFileOrDie(mean_file.c_str(), &blob_proto);
    data_mean_.FromProto(blob_proto);
  }
  // check if we want to use mean_value
  if (transform_param_.mean_value_size() > 0) {
    CHECK(transform_param_.has_mean_file() == false) <<
      "Cannot specify mean_file and mean_value at the same time";
    CHECK_GE(transform_param_.mean_value_size(), bottom[0]->channels());
    for (int c = 0; c < transform_param_.mean_value_size(); ++c) {
      mean_values_.push_back(transform_param_.mean_value(c));
    }
  }
  //xlen_ = std::ceil(std::sqrt(bottom[0]->num()));
  //ylen_ = bottom[0]->num()/xlen_;
  xlen_ = 4;
  ylen_ = 4;
  wholewidth_ = 2*xlen_*width_+1; 
  wholeheight_ = ylen_*height_+1;
  decoded_image_ = cv::Mat::zeros(wholeheight_, wholewidth_, CV_8UC3);
  images_.resize(bottom[0]->num());
  for(int k=0;k<images_.size();k++)
    images_[k] = cv::Mat::zeros(height_, 2*width_, CV_8UC3);
  iter_ = 0;
}

template <typename Dtype>
void VisualizeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  vector<int> top_shape(0);  // Accuracy is a scalar; 0 axes.
  top[0]->Reshape(top_shape);
}

template <typename Dtype>
void VisualizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
//LOG(INFO) <<"vIS" << iter_ << " "<<iter_ % display_interval_;
  if(iter_ % display_interval_ == 0){
	//LOG(INFO) <<"vIS";
    const Dtype *ori_bottom_data = bottom[0]->cpu_data();
    const Dtype *con_bottom_data = bottom[1]->cpu_data();
    const int dim = bottom[0]->count()/bottom[0]->num();
    CHECK_EQ(bottom[0]->channels(), 3);
    CHECK_EQ(bottom[0]->channels(),images_[0].channels());
    CHECK_EQ(dim,bottom[0]->channels()*bottom[0]->width()*bottom[0]->height());
    int channels = 3;
    //if(gray_)channels=1;
    
    for(int k=0;k<images_.size();k++){
      unsigned char* input =(unsigned char*)(images_[k].data);
      //LOG(INFO)<<images_[k].rows<<" "<<images_[k].cols<<" "<<images_[k].channels();
      for(int i=0;i<height_;i++){
        for(int j=0;j<width_;j++){
          if(transform_param_.mean_value_size() > 0){
			Dtype ori_b = (ori_bottom_data[k*dim+ 0*width_*height_+i*width_+j] + mean_values_[0]) * scale_;
			ori_b = std::max(ori_b,Dtype(0.0)); ori_b = std::min(ori_b,Dtype(255.0));
			Dtype ori_g = (ori_bottom_data[k*dim+ 1*width_*height_+i*width_+j] + mean_values_[1]) * scale_;
			ori_g = std::max(ori_g,Dtype(0.0)); ori_g = std::min(ori_g,Dtype(255.0));
			Dtype ori_r = (ori_bottom_data[k*dim+ 2*width_*height_+i*width_+j] + mean_values_[2]) * scale_;
			ori_r = std::max(ori_r,Dtype(0.0)); ori_r = std::min(ori_r,Dtype(255.0));
			Dtype con_b = (con_bottom_data[k*dim+ 0*width_*height_+i*width_+j] + mean_values_[0]) * scale_;
			con_b = std::max(con_b,Dtype(0.0)); con_b = std::min(con_b,Dtype(255.0));
            Dtype con_g = (con_bottom_data[k*dim+ 1*width_*height_+i*width_+j] + mean_values_[1]) * scale_;
			con_g = std::max(con_g,Dtype(0.0)); con_g = std::min(con_g,Dtype(255.0));
            Dtype con_r = (con_bottom_data[k*dim+ 2*width_*height_+i*width_+j] + mean_values_[2]) * scale_;
			con_r = std::max(con_r,Dtype(0.0)); con_r = std::min(con_r,Dtype(255.0));
            input[i*(2*width_)*channels+j*channels+0] = ori_b;
            input[i*(2*width_)*channels+(j+width_)*channels+0] = con_b;
            input[i*(2*width_)*channels+j*channels+1] = ori_g;
            input[i*(2*width_)*channels+(j+width_)*channels+1] = con_g;
            input[i*(2*width_)*channels+j*channels+2] = ori_r;
            input[i*(2*width_)*channels+(j+width_)*channels+2] = con_r;
          }else if(transform_param_.has_mean_file()){
            Dtype* mean = data_mean_.mutable_cpu_data();
            int x_offset = (data_mean_.width() - width_) / 2;
            int y_offset = (data_mean_.height() - height_) / 2;
            int mean_b = mean[(0 * data_mean_.height() + y_offset + i) * data_mean_.width() + x_offset + j];
            int mean_g = mean[(1 * data_mean_.height() + y_offset + i) * data_mean_.width() + x_offset + j];
            int mean_r = mean[(2 * data_mean_.height() + y_offset + i) * data_mean_.width() + x_offset + j];
            input[i*(2*width_)*channels+j*channels+0] = (ori_bottom_data[k*dim+ 0*width_*height_+i*width_+j] + mean_b) * scale_;
            input[i*(2*width_)*channels+(j+width_)*channels+0] = (con_bottom_data[k*dim+ 0*width_*height_+i*width_+j] + mean_b) * scale_;
            input[i*(2*width_)*channels+j*channels+1] = (ori_bottom_data[k*dim+ 1*width_*height_+i*width_+j] + mean_g) * scale_;
            input[i*(2*width_)*channels+(j+width_)*channels+1] = (con_bottom_data[k*dim+ 1*width_*height_+i*width_+j] + mean_g) * scale_;
            input[i*(2*width_)*channels+j*channels+2] = (ori_bottom_data[k*dim+ 2*width_*height_+i*width_+j] + mean_r) * scale_;
            input[i*(2*width_)*channels+(j+width_)*channels+2] = (con_bottom_data[k*dim+ 2*width_*height_+i*width_+j] + mean_r) * scale_;
          }else{
            input[i*(2*width_)*channels+j*channels+0]=ori_bottom_data[k*dim+ 0*width_*height_+i*width_+j]*scale_;
            input[i*(2*width_)*channels+(j+width_)*channels+0]=con_bottom_data[k*dim+ 0*width_*height_+i*width_+j]*scale_;
            input[i*(2*width_)*channels+j*channels+1]=ori_bottom_data[k*dim+ 1*width_*height_+i*width_+j]*scale_;
            input[i*(2*width_)*channels+(j+width_)*channels+1]=con_bottom_data[k*dim+ 1*width_*height_+i*width_+j]*scale_;
            input[i*(2*width_)*channels+j*channels+2]=ori_bottom_data[k*dim+ 2*width_*height_+i*width_+j]*scale_;
            input[i*(2*width_)*channels+(j+width_)*channels+2]=con_bottom_data[k*dim+ 2*width_*height_+i*width_+j]*scale_;
          }
        }
      }
      if(k<16){
        cv::Rect roi = cv::Rect((k%xlen_)*2*width_,(k/xlen_)*height_,2*width_,height_);
        cv::resize(images_[k], decoded_image_(roi), images_[k].size());
      }
    }
    //cv::imshow("decoded images", decoded_image_);
    std::string decoded_image_path = root_folder_ + "/" + std::to_string(iter_)+".jpg";
    cv::imwrite(decoded_image_path.c_str(), decoded_image_);
    //cv::waitKey(1);
  }
  iter_++;
}

template <typename Dtype>
void VisualizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

INSTANTIATE_CLASS(VisualizeLayer);
REGISTER_LAYER_CLASS(Visualize);

}  // namespace caffe
