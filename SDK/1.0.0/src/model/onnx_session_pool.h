#ifndef WENET_SDK_SRC_MODEL_ONNX_SESSION_POOL_H_
#define WENET_SDK_SRC_MODEL_ONNX_SESSION_POOL_H_

#include <memory>
#include <mutex>
#include <vector>

#include "model/onnx_ctc_model.h"

namespace wenet_sdk::internal {

class OnnxSessionPool {
 public:
  void Add(std::unique_ptr<OnnxCtcModel> model);
  std::unique_ptr<OnnxCtcModel> Take();
  void Put(std::unique_ptr<OnnxCtcModel> model);
  size_t size() const;

 private:
  mutable std::mutex mutex_;
  std::vector<std::unique_ptr<OnnxCtcModel>> models_;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_MODEL_ONNX_SESSION_POOL_H_
