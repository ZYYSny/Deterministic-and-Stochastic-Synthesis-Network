#include <vector>

#include "caffe/layers/base_data_layer.hpp"

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  if (prefetch_current_) {
    prefetch_free_.push(prefetch_current_);
  }
  prefetch_current_ = prefetch_full_.pop("Waiting for data");
  // Reshape to loaded data.
  top[0]->ReshapeLike(prefetch_current_->data_);
  top[0]->set_gpu_data(prefetch_current_->data_.mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(prefetch_current_->label_);
    top[1]->set_gpu_data(prefetch_current_->label_.mutable_gpu_data());
  }
  if (this->output_edges_) {
    // Reshape to loaded edges.
    top[2]->ReshapeLike(prefetch_current_->edge_);
    top[2]->set_gpu_data(prefetch_current_->edge_.mutable_gpu_data());
  }
  if (this->output_dims_) {
    // Reshape to loaded edges.
    top[3]->ReshapeLike(prefetch_current_->dim_);
    top[3]->set_gpu_data(prefetch_current_->dim_.mutable_gpu_data());
  }
  if (this->output_textures_) {
    // Reshape to loaded dims.
    top[4]->ReshapeLike(prefetch_current_->texture_);
    top[4]->set_cpu_data(prefetch_current_->texture_.mutable_gpu_data());
  }
}

INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);

}  // namespace caffe
