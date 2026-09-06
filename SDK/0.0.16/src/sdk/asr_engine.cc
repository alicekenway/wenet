#include "asr_sdk/asr_engine.h"

#include <exception>
#include <filesystem>
#include <iostream>

#include "flashlight_decoder/flashlight_asr_stream.h"
#include "flashlight_decoder/compiled_decode_context.h"
#include "flashlight_decoder/flashlight_decoder_resource.h"
#include "itn/itn_processor.h"
#include "package/model_package.h"
#include "package/model_package_validator.h"
#include "sherpa_onnx_wenet/ctc_onnx_backend_factory.h"
#include "sherpa_onnx_wenet/streaming_ctc_backend.h"
#if !ASR_SDK_PRODUCTION_FLASHLIGHT_ONLY
#include "sdk/asr_stream_internal.h"
#include "wenet_bridge/wenet_runtime_bridge.h"
#endif

namespace asr_sdk {
namespace {

#if !ASR_SDK_PRODUCTION_FLASHLIGHT_ONLY
struct NonFlashlightContextOwnerToken {};

class EmptyDecodeContext final : public DecodeContext {
 public:
  explicit EmptyDecodeContext(
      std::shared_ptr<const NonFlashlightContextOwnerToken> owner)
      : owner_(std::move(owner)) {}

  bool BelongsTo(
      const std::shared_ptr<const NonFlashlightContextOwnerToken>& owner) const {
    return owner_ == owner;
  }

 private:
  std::shared_ptr<const NonFlashlightContextOwnerToken> owner_;
};

class AsrEngineImpl final : public AsrEngine {
 public:
  AsrEngineImpl(EngineConfig config,
                std::unique_ptr<internal::WenetRuntimeBridge> bridge,
                std::shared_ptr<const internal::ItnProcessor> itn)
      : config_(std::move(config)), bridge_(std::move(bridge)),
        itn_(std::move(itn)) {}

  StatusOr<std::shared_ptr<const DecodeContext>> CompileDecodeContext(
      const DecodeContextConfig& config) override {
    if (!config.contacts.empty() || !config.slots.empty()) {
      return Status::FailedPrecondition(
          "runtime contacts require a Flashlight contact-ready package");
    }
    return std::shared_ptr<const DecodeContext>(
        std::make_shared<EmptyDecodeContext>(context_owner_));
  }

  StatusOr<std::unique_ptr<AsrStream>> CreateStream(
      std::shared_ptr<const DecodeContext> context) override {
    if (!context) {
      return CreateStream();
    }
    const auto empty = std::dynamic_pointer_cast<const EmptyDecodeContext>(context);
    if (!empty || !empty->BelongsTo(context_owner_)) {
      return Status::FailedPrecondition(
          "decode context was compiled by a different engine");
    }
    return CreateStream();
  }

  StatusOr<std::unique_ptr<AsrStream>> CreateStream() override {
    auto adapter_or = bridge_->CreateStream();
    if (!adapter_or.ok()) {
      return adapter_or.status();
    }
    return internal::WrapWithItn(
        internal::MakeAsrStream(std::move(adapter_or).value()), itn_);
  }

  const EngineConfig& config() const override { return config_; }

 private:
  EngineConfig config_;
  std::unique_ptr<internal::WenetRuntimeBridge> bridge_;
  std::shared_ptr<const internal::ItnProcessor> itn_;
  std::shared_ptr<const NonFlashlightContextOwnerToken> context_owner_ =
      std::make_shared<NonFlashlightContextOwnerToken>();
};
#endif

class FlashlightEngineImpl final : public AsrEngine {
 public:
  FlashlightEngineImpl(
      EngineConfig config,
      std::shared_ptr<internal::flashlight_decoder::FlashlightAsrResources>
          resources,
      std::shared_ptr<const internal::ItnProcessor> itn)
      : config_(std::move(config)), resources_(std::move(resources)),
        itn_(std::move(itn)),
        context_owner_(std::make_shared<
                       internal::flashlight_decoder::ContextOwnerToken>()) {}

  StatusOr<std::shared_ptr<const DecodeContext>> CompileDecodeContext(
      const DecodeContextConfig& config) override {
    auto context_or =
        internal::flashlight_decoder::CompiledDecodeContext::Compile(
            resources_->decoder_resource, config, context_owner_);
    if (!context_or.ok()) {
      return context_or.status();
    }
    if (config_.debug) {
      for (const auto& diagnostic : context_or.value()->Diagnostics()) {
        std::cerr << "dcc: " << diagnostic << '\n';
      }
      const auto& contacts = context_or.value()->DynamicContacts();
      std::cerr << "dcc: compiled_dynamic_forms="
                << (contacts ? contacts->Forms().size() : 0) << '\n';
    }
    return std::shared_ptr<const DecodeContext>(
        std::move(context_or).value());
  }

  StatusOr<std::unique_ptr<AsrStream>> CreateStream(
      std::shared_ptr<const DecodeContext> context) override {
    if (!context) {
      return CreateStream();
    }
    const auto compiled = std::dynamic_pointer_cast<
        const internal::flashlight_decoder::CompiledDecodeContext>(context);
    if (!compiled || !compiled->BelongsTo(context_owner_)) {
      return Status::FailedPrecondition(
          "decode context was compiled by a different engine");
    }
    if (compiled->IsEmpty()) {
      return CreateStream();
    }
    if (!compiled->SlotDecoderResource()) {
      return Status::Internal(
          "compiled decode context has no slot decoder resource");
    }
    return internal::WrapWithItn(std::unique_ptr<AsrStream>(
        new internal::flashlight_decoder::FlashlightAsrStream(
            resources_, compiled->SlotDecoderResource())), itn_);
  }

  StatusOr<std::unique_ptr<AsrStream>> CreateStream() override {
    return internal::WrapWithItn(
        std::unique_ptr<AsrStream>(
            new internal::flashlight_decoder::FlashlightAsrStream(resources_)),
        itn_);
  }

  const EngineConfig& config() const override { return config_; }

 private:
  EngineConfig config_;
  std::shared_ptr<internal::flashlight_decoder::FlashlightAsrResources>
      resources_;
  std::shared_ptr<const internal::ItnProcessor> itn_;
  std::shared_ptr<const internal::flashlight_decoder::ContextOwnerToken>
      context_owner_;
};

StatusOr<std::unique_ptr<AsrEngine>> CreateFlashlightEngine(
    const EngineConfig& config, const internal::ModelPackage& package,
    std::shared_ptr<const internal::ItnProcessor> itn) {
  Status status = internal::ValidateModelPackageLayout(package);
  if (!status.ok()) {
    return status;
  }
  try {
    const internal::flashlight_decoder::FlashlightDecoderOptions options =
        package.flashlight_options;
    if (!package.has_compact_lexicon) {
      return Status::InvalidArgument(
          "SDK 0.0.16 requires decoder_type="
          "flashlight_compact_lexicon_kenlm and lexicon.bin");
    }
    const uint64_t dependency_hash =
        internal::flashlight_decoder::ComputeCompactLexiconDependencyHash(
            package);
    auto decoder_resource =
        std::make_shared<internal::flashlight_decoder::FlashlightDecoderResource>(
            package.tokens_txt, package.words_txt, package.lexicon_bin,
            package.fixed_lms, package.output_mapping_txt,
            package.final_output_mapping_txt, options, package.blank_token,
            package.sil_token, package.unk_word, package.length_penalty, true,
            dependency_hash, package.sentencepiece_model);
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
    resolved.debug = config.debug || package.debug;
    auto shared =
        std::make_shared<internal::flashlight_decoder::FlashlightAsrResources>();
    shared->backend_template =
        std::shared_ptr<internal::sherpa_onnx_wenet::StreamingCtcBackend>(
            std::move(backend_template));
    shared->decoder_resource = std::move(decoder_resource);
    shared->config = resolved;
    shared->feature_type = package.feature_type;
    return std::unique_ptr<AsrEngine>(
        new FlashlightEngineImpl(resolved, std::move(shared), std::move(itn)));
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
  std::shared_ptr<const internal::ItnProcessor> itn;
  if (package.has_itn && config.enable_itn) {
    auto itn_or = internal::CreateItnProcessor(
        package.itn_tagger_fst.string(), package.itn_verbalizer_fst.string());
    if (!itn_or.ok()) return itn_or.status();
    itn = std::move(itn_or).value();
  }
  if (package.has_flashlight_decoder) {
    return CreateFlashlightEngine(config, package, std::move(itn));
  }
#if ASR_SDK_PRODUCTION_FLASHLIGHT_ONLY
  return Status::FailedPrecondition(
      "this Android SDK supports the production Flashlight model package only");
#else
  auto bridge_or =
      internal::WenetRuntimeBridge::Create(config, std::move(package));
  if (!bridge_or.ok()) {
    return bridge_or.status();
  }
  auto engine = std::unique_ptr<AsrEngine>(
      new AsrEngineImpl(config, std::move(bridge_or).value(), std::move(itn)));
  return engine;
#endif
}

}  // namespace asr_sdk
