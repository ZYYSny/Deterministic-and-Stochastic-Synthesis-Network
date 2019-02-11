#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <math.h>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/random_patch_crop_resize_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
RandomPatchCropResizeLayer<Dtype>::~RandomPatchCropResizeLayer<Dtype>() {
  this->StopInternalThread();
}

template <typename Dtype>
void RandomPatchCropResizeLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  const bool is_color  = this->layer_param_.image_data_param().is_color();
  lower_bound_ = this->layer_param_.random_crop_param().lower_bound();
  upper_bound_ = this->layer_param_.random_crop_param().upper_bound();
  CHECK_EQ(this->layer_param_.transform_param().crop_size(),new_height);
  CHECK_EQ(this->layer_param_.transform_param().crop_size(),new_width);
  scale_ =  this->layer_param_.bilinear_interp_param().scale();
  label_height_ = round(scale_ * new_height);
  label_width_ = round(scale_ * new_width);
  string root_folder = this->layer_param_.image_data_param().root_folder();
  CHECK_GT(upper_bound_,lower_bound_);
  CHECK_GE(lower_bound_,new_width);
  CHECK_GE(lower_bound_,new_height);
  InitRand();
  
  CHECK((new_height == 0 && new_width == 0) ||
      (new_height > 0 && new_width > 0)) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename, label;
  while (infile >> filename >> label) {
    lines_.push_back(std::make_pair(filename, label));
  }

  if (this->layer_param_.image_data_param().shuffle()) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";
  
  std::ofstream outputpath("Oxford_Flowers_Path.txt");
  for(int k=0;k<lines_.size();k++){
      outputpath<<lines_[k].first<<"\t"<<lines_[k].second<<std::endl;
  }
  
  lines_id_ = 0;
  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }
  // Read an image, and use it to initialize the top blob.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_image.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape prefetch_data and top[0] according to the batch_size.
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  top_shape[0] = batch_size;
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);

  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // label
  //vector<int> label_shape(1, batch_size);
  top[1]->Reshape({batch_size,top[0]->channels(),label_height_,label_width_});
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape({top[1]->num(),top[1]->channels(),top[1]->height(),top[1]->width()});
  }
  LOG(INFO) << "output label size: " << top[1]->num() << ","
      << top[1]->channels() << "," << top[1]->height() << ","
      << top[1]->width(); 
}

template <typename Dtype>
void RandomPatchCropResizeLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void RandomPatchCropResizeLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  CHECK(this->transformed_data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  const bool is_color = image_data_param.is_color();
  string root_folder = image_data_param.root_folder();

  // Reshape according to the first image of each batch
  // on single input batches allows for inputs of varying dimension.
  cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
      new_height, new_width, is_color);
  CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
  // Use data_transformer to infer the expected blob shape from a cv_img.
  vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
  this->transformed_data_.Reshape(top_shape);
  // Reshape batch according to the batch_size.
  top_shape[0] = batch_size;
  batch->data_.Reshape(top_shape);

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // get a blob
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    //cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
    //    new_height, new_width, is_color);
    cv::Mat raw_img = cv::imread(root_folder + lines_[lines_id_].first, is_color);
    CHECK(raw_img.data) << "Could not load " << lines_[lines_id_].first;
	while(raw_img.cols<=new_width||raw_img.rows<=new_height){
		lines_id_++;
		raw_img = cv::imread(root_folder + lines_[lines_id_].first, is_color);
		LOG(FATAL)<<"!!!!";
	}
    cv::Mat val_img = cv::imread(root_folder + lines_[lines_id_].second, is_color);
    CHECK_EQ(round(val_img.cols/scale_),raw_img.cols);
	CHECK_EQ(round(val_img.rows/scale_),raw_img.rows);
    int raw_image_width = raw_img.cols;
    int raw_image_height = raw_img.rows;
    //LOG(INFO)<<upper_bound_<<"\t"<<lower_bound_;
    int interval=upper_bound_-lower_bound_;
    int randomized_short_side = Rand(interval)+lower_bound_;
    /*if(raw_image_width>=raw_image_height){
      raw_image_width = randomized_short_side * raw_image_width / Dtype(raw_image_height);
      raw_image_height = randomized_short_side;
    }else{
      raw_image_height = randomized_short_side * raw_image_height / Dtype(raw_image_width);
      raw_image_width = randomized_short_side;    
    }
    cv::resize(raw_img,raw_img,cv::Size(raw_image_width,raw_image_height));*/
    int x_anchor = Rand(raw_image_width-new_width+1);
    int y_anchor = Rand(raw_image_height-new_height+1);
    if(x_anchor%24 != 0){
	   x_anchor = 24*(x_anchor/24);
	}
	CHECK_EQ(x_anchor%24, 0);
    if(y_anchor%24 != 0){
	   y_anchor = 24*(y_anchor/24);
	}
	CHECK_EQ(y_anchor%24, 0);
    cv::Mat cv_img;
    cv_img = raw_img(cv::Rect(x_anchor,y_anchor,new_width,new_height));
    read_time += timer.MicroSeconds();
    timer.Start();
	//
	int xx_anchor = round(x_anchor*scale_);
	int yy_anchor = round(y_anchor*scale_);
    //LOG(INFO) << xx_anchor << "\t" << x_anchor << "\t" << yy_anchor << "\t" << y_anchor;
	cv::Mat rz_img = val_img(cv::Rect(xx_anchor,yy_anchor,label_width_,label_height_));
    // Apply transformations (mirror, crop...) to the image
    //int offset = batch->data_.offset(item_id);
    //this->transformed_data_.set_cpu_data(prefetch_data + offset);
    //this->data_transformer_->Transform(cv_img, &(this->transformed_data_));
    //trans_time += timer.MicroSeconds();
    //read data
    int data_type = Rand(6);
	CHECK_EQ(new_height, new_width);
    if (data_type == 0){
      for(int h = 0; h < new_height; h++){
	    for(int w = 0; w < new_width; w++){
	      prefetch_data[item_id*3*new_height*new_width+0*new_height*new_width+h*new_width+w] = cv_img.at<cv::Vec3b>(h,w)[0];
		  prefetch_data[item_id*3*new_height*new_width+1*new_height*new_width+h*new_width+w] = cv_img.at<cv::Vec3b>(h,w)[1];
		  prefetch_data[item_id*3*new_height*new_width+2*new_height*new_width+h*new_width+w] = cv_img.at<cv::Vec3b>(h,w)[2];
	    }
	  }
	  for(int h = 0; h < label_height_; h++){
	    for(int w = 0; w < label_width_; w++){
	      prefetch_label[item_id*3*label_height_*label_width_+0*label_height_*label_width_+h*label_width_+w] = rz_img.at<cv::Vec3b>(h,w)[0];
	      prefetch_label[item_id*3*label_height_*label_width_+1*label_height_*label_width_+h*label_width_+w] = rz_img.at<cv::Vec3b>(h,w)[1];
	      prefetch_label[item_id*3*label_height_*label_width_+2*label_height_*label_width_+h*label_width_+w] = rz_img.at<cv::Vec3b>(h,w)[2];
	    }
	  }	
	}else if (data_type == 1){
      for(int h = 0; h < new_height; h++){
	    for(int w = 0; w < new_width; w++){
	      prefetch_data[item_id*3*new_height*new_width+0*new_height*new_width+h*new_width+w] = cv_img.at<cv::Vec3b>(w,h)[0];
		  prefetch_data[item_id*3*new_height*new_width+1*new_height*new_width+h*new_width+w] = cv_img.at<cv::Vec3b>(w,h)[1];
		  prefetch_data[item_id*3*new_height*new_width+2*new_height*new_width+h*new_width+w] = cv_img.at<cv::Vec3b>(w,h)[2];
	    }
	  }
	  for(int h = 0; h < label_height_; h++){
	    for(int w = 0; w < label_width_; w++){
	      prefetch_label[item_id*3*label_height_*label_width_+0*label_height_*label_width_+h*label_width_+w] = rz_img.at<cv::Vec3b>(w,h)[0];
	      prefetch_label[item_id*3*label_height_*label_width_+1*label_height_*label_width_+h*label_width_+w] = rz_img.at<cv::Vec3b>(w,h)[1];
	      prefetch_label[item_id*3*label_height_*label_width_+2*label_height_*label_width_+h*label_width_+w] = rz_img.at<cv::Vec3b>(w,h)[2];
	    }
	  }
	}else if (data_type == 2){
      for(int h = 0; h < new_height; h++){
	    for(int w = 0; w < new_width; w++){
	      prefetch_data[item_id*3*new_height*new_width+0*new_height*new_width+h*new_width+w] = cv_img.at<cv::Vec3b>(h, new_width-1-w)[0];
		  prefetch_data[item_id*3*new_height*new_width+1*new_height*new_width+h*new_width+w] = cv_img.at<cv::Vec3b>(h, new_width-1-w)[1];
		  prefetch_data[item_id*3*new_height*new_width+2*new_height*new_width+h*new_width+w] = cv_img.at<cv::Vec3b>(h, new_width-1-w)[2];
	    }
	  }
	  for(int h = 0; h < label_height_; h++){
	    for(int w = 0; w < label_width_; w++){
	      prefetch_label[item_id*3*label_height_*label_width_+0*label_height_*label_width_+h*label_width_+w] = rz_img.at<cv::Vec3b>(h, label_width_-1-w)[0];
	      prefetch_label[item_id*3*label_height_*label_width_+1*label_height_*label_width_+h*label_width_+w] = rz_img.at<cv::Vec3b>(h, label_width_-1-w)[1];
	      prefetch_label[item_id*3*label_height_*label_width_+2*label_height_*label_width_+h*label_width_+w] = rz_img.at<cv::Vec3b>(h, label_width_-1-w)[2];
	    }
	  }
	}else if (data_type == 3){
      for(int h = 0; h < new_height; h++){
	    for(int w = 0; w < new_width; w++){
	      prefetch_data[item_id*3*new_height*new_width+0*new_height*new_width+h*new_width+w] = cv_img.at<cv::Vec3b>(new_height-1-h,w)[0];
		  prefetch_data[item_id*3*new_height*new_width+1*new_height*new_width+h*new_width+w] = cv_img.at<cv::Vec3b>(new_height-1-h,w)[1];
		  prefetch_data[item_id*3*new_height*new_width+2*new_height*new_width+h*new_width+w] = cv_img.at<cv::Vec3b>(new_height-1-h,w)[2];
	    }
	  }
	  for(int h = 0; h < label_height_; h++){
	    for(int w = 0; w < label_width_; w++){
	      prefetch_label[item_id*3*label_height_*label_width_+0*label_height_*label_width_+h*label_width_+w] = rz_img.at<cv::Vec3b>(label_height_-1-h,w)[0];
	      prefetch_label[item_id*3*label_height_*label_width_+1*label_height_*label_width_+h*label_width_+w] = rz_img.at<cv::Vec3b>(label_height_-1-h,w)[1];
	      prefetch_label[item_id*3*label_height_*label_width_+2*label_height_*label_width_+h*label_width_+w] = rz_img.at<cv::Vec3b>(label_height_-1-h,w)[2];
	    }
	  }
	}else if (data_type == 4){
      for(int h = 0; h < new_height; h++){
	    for(int w = 0; w < new_width; w++){
	      prefetch_data[item_id*3*new_height*new_width+0*new_height*new_width+h*new_width+w] = cv_img.at<cv::Vec3b>(new_height-1-h, new_width-1-w)[0];
		  prefetch_data[item_id*3*new_height*new_width+1*new_height*new_width+h*new_width+w] = cv_img.at<cv::Vec3b>(new_height-1-h, new_width-1-w)[1];
		  prefetch_data[item_id*3*new_height*new_width+2*new_height*new_width+h*new_width+w] = cv_img.at<cv::Vec3b>(new_height-1-h, new_width-1-w)[2];
	    }
	  }
	  for(int h = 0; h < label_height_; h++){
	    for(int w = 0; w < label_width_; w++){
	      prefetch_label[item_id*3*label_height_*label_width_+0*label_height_*label_width_+h*label_width_+w] = rz_img.at<cv::Vec3b>(label_height_-1-h, label_width_-1-w)[0];
	      prefetch_label[item_id*3*label_height_*label_width_+1*label_height_*label_width_+h*label_width_+w] = rz_img.at<cv::Vec3b>(label_height_-1-h, label_width_-1-w)[1];
	      prefetch_label[item_id*3*label_height_*label_width_+2*label_height_*label_width_+h*label_width_+w] = rz_img.at<cv::Vec3b>(label_height_-1-h, label_width_-1-w)[2];
	    }
	  }
	}else if (data_type == 5){
      for(int h = 0; h < new_height; h++){
	    for(int w = 0; w < new_width; w++){
	      prefetch_data[item_id*3*new_height*new_width+0*new_height*new_width+h*new_width+w] = cv_img.at<cv::Vec3b>(new_width-1-w,new_height-1-h)[0];
		  prefetch_data[item_id*3*new_height*new_width+1*new_height*new_width+h*new_width+w] = cv_img.at<cv::Vec3b>(new_width-1-w,new_height-1-h)[1];
		  prefetch_data[item_id*3*new_height*new_width+2*new_height*new_width+h*new_width+w] = cv_img.at<cv::Vec3b>(new_width-1-w,new_height-1-h)[2];
	    }
	  }
	  for(int h = 0; h < label_height_; h++){
	    for(int w = 0; w < label_width_; w++){
	      prefetch_label[item_id*3*label_height_*label_width_+0*label_height_*label_width_+h*label_width_+w] = rz_img.at<cv::Vec3b>(label_width_-1-w,label_height_-1-h)[0];
	      prefetch_label[item_id*3*label_height_*label_width_+1*label_height_*label_width_+h*label_width_+w] = rz_img.at<cv::Vec3b>(label_width_-1-w,label_height_-1-h)[1];
	      prefetch_label[item_id*3*label_height_*label_width_+2*label_height_*label_width_+h*label_width_+w] = rz_img.at<cv::Vec3b>(label_width_-1-w,label_height_-1-h)[2];
	    }
	  }
	}else{
	   LOG(FATAL)<<"Unkown type "<<data_type;
	}

    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

template <typename Dtype>
void RandomPatchCropResizeLayer<Dtype>::InitRand() {
  const unsigned int rng_seed = caffe_rng_rand();
  rng_.reset(new Caffe::RNG(rng_seed));
}

template <typename Dtype>
int RandomPatchCropResizeLayer<Dtype>::Rand(int n) {
  CHECK(rng_);
  CHECK_GT(n, 0);
  caffe::rng_t* rng =
      static_cast<caffe::rng_t*>(rng_->generator());
  return ((*rng)() % n);
}

INSTANTIATE_CLASS(RandomPatchCropResizeLayer);
REGISTER_LAYER_CLASS(RandomPatchCropResize);

}  // namespace caffe
#endif  // USE_OPENCV