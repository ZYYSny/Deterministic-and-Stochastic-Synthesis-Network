#ifndef CAFFE_RANDOM_PATCH_SKETCH_CROP_DATA_LAYER_HPP_
#define CAFFE_RANDOM_PATCH_SKETCH_CROP_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/**
 * @brief Provides data to the Net from image files.
 *
 * TODO(dox): thorough documentation for Forward and proto params.
 */
template <typename Dtype>
class RandomPatchSketchCropLayer : public BasePrefetchingDataLayer<Dtype> {
 public:
  explicit RandomPatchSketchCropLayer(const LayerParameter& param)
      : BasePrefetchingDataLayer<Dtype>(param) {}
  virtual ~RandomPatchSketchCropLayer();
  virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "RandomPatchSketchCrop"; }
  virtual inline int ExactNumBottomBlobs() const { return 0; }
  virtual inline int ExactNumTopBlobs() const { return 5; }

 protected:
  shared_ptr<Caffe::RNG> prefetch_rng_;
  virtual void ShuffleImages();
  virtual void load_batch(Batch<Dtype>* batch);
  void InitRand();
  int Rand(int n);

  vector<std::pair<std::pair<std::pair<cv::Mat, cv::Mat>,std::pair<cv::Mat, cv::Mat> >, cv::Mat > > lines_;
  vector<std::pair<std::pair<std::pair<cv::string, cv::string>,std::pair<cv::string, cv::string> >, cv::string> > paths_;
  int lines_id_;
  int lower_bound_;
  int upper_bound_;
  shared_ptr<Caffe::RNG> rng_;
  Dtype scale_;
  int label_height_;
  int label_width_;
};


}  // namespace caffe

#endif  // CAFFE_RANDOM_PATCH_SKETCH_CROP_DATA_LAYER_HPP_