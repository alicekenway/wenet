#ifndef ASR_SDK_SRC_WENET_BRIDGE_WENET_RUNTIME_BRIDGE_H_
#define ASR_SDK_SRC_WENET_BRIDGE_WENET_RUNTIME_BRIDGE_H_

#include <memory>

#include "asr_sdk/config.h"
#include "asr_sdk/status.h"
#include "package/model_package.h"
#include "wenet_bridge/wenet_stream_adapter.h"

namespace asr_sdk::internal {

namespace wenet_types {
struct Shared;
}  // namespace wenet_types

class WenetRuntimeBridge {
 public:
  static StatusOr<std::unique_ptr<WenetRuntimeBridge>> Create(
      const EngineConfig& config, ModelPackage package);

  StatusOr<std::unique_ptr<WenetStreamAdapter>> CreateStream();
  const ModelPackage& package() const { return package_; }

 private:
  WenetRuntimeBridge(EngineConfig config, ModelPackage package);

  EngineConfig config_;
  ModelPackage package_;
  std::shared_ptr<wenet_types::Shared> shared_;
};

}  // namespace asr_sdk::internal

#endif  // ASR_SDK_SRC_WENET_BRIDGE_WENET_RUNTIME_BRIDGE_H_
