#include "asr_sdk/asr_engine.h"

#include <exception>

#include "flashlight_decoder/flashlight_asr_stream.h"
#include "flashlight_decoder/flashlight_decoder_resource.h"
#include "package/model_package.h"
#include "package/model_package_validator.h"
#include "sdk/asr_stream_internal.h"
#include "sherpa_onnx_wenet/ctc_onnx_backend_factory.h"
#include "sherpa_onnx_wenet/streaming_ctc_backend.h"
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

class FlashlightEngineImpl final : public AsrEngine {
 public:
  FlashlightEngineImpl(
      EngineConfig config,
      std::shared_ptr<internal::flashlight_decoder::FlashlightAsrResources>
          resources)
      : config_(std::move(config)), resources_(std::move(resources)) {}

  StatusOr<std::unique_ptr<AsrStream>> CreateStream() override {
    return std::unique_ptr<AsrStream>(
        new internal::flashlight_decoder::FlashlightAsrStream(resources_));
  }

  const EngineConfig& config() const override { return config_; }

 private:
  EngineConfig config_;
  std::shared_ptr<internal::flashlight_decoder::FlashlightAsrResources>
      resources_;
};

StatusOr<std::unique_ptr<AsrEngine>> CreateFlashlightEngine(
    const EngineConfig& config, const internal::ModelPackage& package) {
  Status status = internal::ValidateModelPackage(package);
  if (!status.ok()) {
    return status;
  }
  try {
    const internal::flashlight_decoder::FlashlightDecoderOptions options =
        package.flashlight_options;
    auto decoder_resource =
        std::make_shared<internal::flashlight_decoder::FlashlightDecoderResource>(
            package.tokens_txt, package.words_txt, package.lexicon_txt,
            package.kenlm_bin, package.output_mapping_txt, options,
            package.blank_token, package.sil_token, package.unk_word);
    auto backend_template =
        internal::sherpa_onnx_wenet::CreateStreamingCtcBackend(
            package.sherpa_ctc_onnx.string(), config.num_threads,
            decoder_resource->BlankId());
    if (backend_template->Info().vocab_size !=
        decoder_resource->AmTokens().ModelVocabSize()) {
      return Status::InvalidArgument(
          "ONNX vocab size does not match tokens.txt model vocab size");
    }

    EngineConfig resolved = config;
    resolved.model_dir = package.root.string();
    resolved.sample_rate = package.sample_rate;
    resolved.nbest = package.nbest;
    auto shared =
        std::make_shared<internal::flashlight_decoder::FlashlightAsrResources>();
    shared->backend_template =
        std::shared_ptr<internal::sherpa_onnx_wenet::StreamingCtcBackend>(
            std::move(backend_template));
    shared->decoder_resource = std::move(decoder_resource);
    shared->config = resolved;
    shared->feature_type = package.feature_type;
    return std::unique_ptr<AsrEngine>(
        new FlashlightEngineImpl(resolved, std::move(shared)));
  } catch (const std::exception& e) {
    return Status::Internal(std::string("failed to load Flashlight package: ") +
                            e.what());
  }
}

}  // namespace

StatusOr<std::unique_ptr<AsrEngine>> AsrEngine::Create(
    const EngineConfig& config) {
  auto package_or = internal::LoadModelPackage(config);
  if (!package_or.ok()) {
    return package_or.status();
  }
  internal::ModelPackage package = std::move(package_or).value();
  if (package.has_flashlight_decoder) {
    return CreateFlashlightEngine(config, package);
  }
  auto bridge_or =
      internal::WenetRuntimeBridge::Create(config, std::move(package));
  if (!bridge_or.ok()) {
    return bridge_or.status();
  }
  auto engine = std::unique_ptr<AsrEngine>(
      new AsrEngineImpl(config, std::move(bridge_or).value()));
  return engine;
}

}  // namespace asr_sdk
