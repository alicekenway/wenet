#include "asr_sdk/asr_engine.h"

#include "package/model_package.h"
#include "sdk/asr_stream_internal.h"
#include "wenet_bridge/wenet_runtime_bridge.h"

namespace asr_sdk {
namespace {

class AsrEngineImpl final : public AsrEngine {
 public:
  AsrEngineImpl(EngineConfig config,
                std::unique_ptr<internal::WenetRuntimeBridge> bridge)
      : config_(std::move(config)), bridge_(std::move(bridge)) {}

  StatusOr<std::unique_ptr<AsrStream>> CreateStream() override {
    auto adapter_or = bridge_->CreateStream();
    if (!adapter_or.ok()) {
      return adapter_or.status();
    }
    return internal::MakeAsrStream(std::move(adapter_or).value());
  }

  const EngineConfig& config() const override { return config_; }

 private:
  EngineConfig config_;
  std::unique_ptr<internal::WenetRuntimeBridge> bridge_;
};

}  // namespace

StatusOr<std::unique_ptr<AsrEngine>> AsrEngine::Create(
    const EngineConfig& config) {
  auto package_or = internal::LoadModelPackage(config);
  if (!package_or.ok()) {
    return package_or.status();
  }
  auto bridge_or =
      internal::WenetRuntimeBridge::Create(config, std::move(package_or).value());
  if (!bridge_or.ok()) {
    return bridge_or.status();
  }
  auto engine = std::unique_ptr<AsrEngine>(
      new AsrEngineImpl(config, std::move(bridge_or).value()));
  return engine;
}

}  // namespace asr_sdk
