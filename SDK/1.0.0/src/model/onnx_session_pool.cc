#include "model/onnx_session_pool.h"

namespace wenet_sdk::internal {

void OnnxSessionPool::Add(std::unique_ptr<OnnxCtcModel> model) {
  Put(std::move(model));
}

std::unique_ptr<OnnxCtcModel> OnnxSessionPool::Take() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (models_.empty()) {
    return nullptr;
  }
  auto model = std::move(models_.back());
  models_.pop_back();
  return model;
}

void OnnxSessionPool::Put(std::unique_ptr<OnnxCtcModel> model) {
  if (!model) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  models_.push_back(std::move(model));
}

size_t OnnxSessionPool::size() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return models_.size();
}

}  // namespace wenet_sdk::internal
