#include "wenet_sdk/asr_engine.h"

#include <memory>

#include "core/engine_resources.h"
#include "core/stream_session.h"
#include "decoder/fst_loader.h"
#include "model/model_metadata.h"
#include "wenet_sdk/version.h"

namespace wenet_sdk {
namespace {

class AsrEngineImpl final : public AsrEngine {
 public:
  explicit AsrEngineImpl(std::shared_ptr<internal::EngineResources> resources)
      : resources_(std::move(resources)) {}

  std::unique_ptr<Stream> CreateStream() override {
    auto stream = std::make_unique<internal::StreamSession>(resources_);
    auto status = stream->Init();
    if (!status.ok()) {
      return nullptr;
    }
    return stream;
  }

 private:
  std::shared_ptr<internal::EngineResources> resources_;
};

}  // namespace

std::unique_ptr<AsrEngine> AsrEngine::Create(const EngineConfig& config) {
  auto resources = std::make_shared<internal::EngineResources>();
  resources->config = config;

  auto status =
      internal::LoadModelMetadata(config.model_dir, &resources->metadata);
  if (!status.ok()) {
    return nullptr;
  }

  if (resources->config.chunk_size <= 0) {
    resources->config.chunk_size = resources->metadata.streaming.chunk_size;
  }
  if (resources->config.num_left_chunks <= 0) {
    resources->config.num_left_chunks =
        resources->metadata.streaming.num_left_chunks;
  }
  if (resources->config.blank_skip_thresh <= 0.0f) {
    resources->config.blank_skip_thresh =
        resources->metadata.decoder.blank_skip_thresh;
  }
  if (resources->config.waveform_scale <= 0.0f) {
    resources->config.waveform_scale = resources->metadata.waveform_scale;
  }

  status = internal::ValidateModelPackageFiles(resources->metadata, true);
  if (!status.ok()) {
    return nullptr;
  }
  status = resources->tokens.Load(
      resources->metadata.Resolve(resources->metadata.vocab.tokens));
  if (!status.ok()) {
    return nullptr;
  }
  status = resources->words.Load(
      resources->metadata.Resolve(resources->metadata.vocab.words));
  if (!status.ok()) {
    return nullptr;
  }
  if (!resources->tokens.Contains(resources->metadata.vocab.blank_id)) {
    return nullptr;
  }
  if (resources->metadata.decoder.type == DecoderType::kCtcWfst) {
    status = internal::ValidateFstCompatibility(
        resources->metadata.Resolve(resources->metadata.decoder.graph),
        resources->tokens.Size(), resources->words.Size());
    if (!status.ok()) {
      return nullptr;
    }
  }

  return std::make_unique<AsrEngineImpl>(std::move(resources));
}

const char* VersionString() {
  return "0.1.0";
}

int AbiVersion() {
  return WENET_SDK_ABI_VERSION;
}

}  // namespace wenet_sdk
