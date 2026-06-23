#include "core/recognizer.h"

#include <algorithm>

#include "decoder/ctc_prefix_decoder.h"
#include "decoder/ctc_wfst_decoder.h"
#include "decoder/greedy_ctc_decoder.h"

namespace wenet_sdk::internal {

namespace {

FeaturePipelineConfig ToFeatureConfig(const EngineResources& resources) {
  FeaturePipelineConfig config;
  config.sample_rate = resources.metadata.sample_rate;
  config.feature_dim = resources.metadata.feature_dim;
  config.frame_length_ms = resources.metadata.frame_length_ms;
  config.frame_shift_ms = resources.metadata.frame_shift_ms;
  config.waveform_scale = resources.config.waveform_scale > 0.0f
                              ? resources.config.waveform_scale
                              : resources.metadata.waveform_scale;
  config.chunk_size = resources.config.chunk_size > 0
                          ? resources.config.chunk_size
                          : resources.metadata.streaming.chunk_size;
  return config;
}

BlankSkipperOptions ToBlankSkipperOptions(const EngineResources& resources) {
  BlankSkipperOptions options;
  options.blank_id = resources.metadata.vocab.blank_id;
  options.blank_skip_thresh = resources.config.blank_skip_thresh;
  options.enabled = resources.config.blank_skip_thresh <= 1.0f;
  return options;
}

EndpointOptions ToEndpointOptions(const EngineResources& resources) {
  EndpointOptions options;
  options.endpoint_silence_ms = resources.metadata.streaming.endpoint_silence_ms;
  options.max_segment_ms = resources.metadata.streaming.max_segment_ms;
  options.frame_shift_ms =
      resources.metadata.frame_shift_ms * resources.metadata.subsampling_rate;
  return options;
}

TextNormalizerOptions ToTextOptions(const EngineResources& resources) {
  TextNormalizerOptions options;
  options.lowercase = resources.metadata.postprocess.lowercase;
  options.remove_bpe_marker = resources.metadata.postprocess.remove_bpe_marker;
  options.language_type = resources.metadata.postprocess.language_type;
  return options;
}

}  // namespace

Recognizer::Recognizer(std::shared_ptr<const EngineResources> resources)
    : resources_(std::move(resources)),
      feature_pipeline_(ToFeatureConfig(*resources_)),
      model_(resources_->metadata, resources_->tokens.Size()),
      blank_skipper_(ToBlankSkipperOptions(*resources_)),
      endpoint_(ToEndpointOptions(*resources_)),
      result_builder_(
          &resources_->tokens, &resources_->words,
          TextNormalizer(ToTextOptions(*resources_)),
          TimestampEstimator(resources_->metadata.frame_shift_ms,
                             resources_->metadata.subsampling_rate),
          resources_->config.enable_timestamps) {}

Status Recognizer::Init() {
  auto status = feature_pipeline_.LoadCmvnIfPresent(
      resources_->metadata.Resolve("global_cmvn"));
  if (!status.ok()) {
    return status;
  }
  status = model_.Init();
  if (!status.ok()) {
    return status;
  }
  decoder_ = CreateDecoder();
  if (!decoder_) {
    return Status::Internal("failed to create decoder");
  }
  return Status::OK();
}

Status Recognizer::AcceptWaveform(int sample_rate, const float* samples,
                                  size_t n) {
  return feature_pipeline_.AcceptWaveform(sample_rate, samples, n);
}

Status Recognizer::AcceptWaveform(int sample_rate, const int16_t* samples,
                                  size_t n) {
  return feature_pipeline_.AcceptWaveform(sample_rate, samples, n);
}

bool Recognizer::DecodeReady() const { return feature_pipeline_.DecodeReady(); }

Status Recognizer::Decode() {
  std::vector<std::vector<float>> features;
  if (!feature_pipeline_.ReadChunk(&features) || features.empty()) {
    return Status::OK();
  }

  std::vector<std::vector<float>> log_probs;
  auto status = model_.ForwardChunk(features, &log_probs);
  if (!status.ok()) {
    return status;
  }

  auto filtered = blank_skipper_.Filter(log_probs);
  const bool blank_heavy = filtered.size() < log_probs.size();
  decoder_->Advance(filtered);
  endpoint_.AdvanceFrames(static_cast<int>(filtered.size()), blank_heavy);

  latest_result_ = result_builder_.Build(decoder_->PartialResult(), false);
  if (resources_->config.enable_endpoint &&
      endpoint_.IsEndpoint(input_finished_ && !feature_pipeline_.DecodeReady())) {
    final_result_ = result_builder_.Build(decoder_->Finalize(), true);
    finalized_ = true;
  }
  return Status::OK();
}

void Recognizer::SetInputFinished() {
  input_finished_ = true;
  feature_pipeline_.SetInputFinished();
}

void Recognizer::Reset() {
  input_finished_ = false;
  finalized_ = false;
  latest_result_ = AsrResult{};
  final_result_ = AsrResult{};
  feature_pipeline_.Reset();
  model_.Reset();
  blank_skipper_.Reset();
  endpoint_.Reset();
  if (decoder_) {
    decoder_->Reset();
  }
}

AsrResult Recognizer::GetFinalResult() {
  if (!finalized_ && decoder_) {
    final_result_ = result_builder_.Build(decoder_->Finalize(), true);
    finalized_ = true;
  }
  return final_result_;
}

std::unique_ptr<StreamingDecoder> Recognizer::CreateDecoder() const {
  DecoderType type = resources_->config.decoder_type;
  if (type == DecoderType::kAuto) {
    type = resources_->metadata.decoder.type;
  }
  if (type == DecoderType::kCtcWfst) {
    CtcWfstDecoderOptions options;
    options.blank_id = resources_->metadata.vocab.blank_id;
    options.beam = resources_->metadata.decoder.beam;
    options.lattice_beam = resources_->metadata.decoder.lattice_beam;
    options.max_active = resources_->metadata.decoder.max_active;
    options.min_active = resources_->metadata.decoder.min_active;
    options.acoustic_scale = resources_->metadata.decoder.acoustic_scale;
    options.lm_scale = resources_->metadata.decoder.lm_scale;
    options.length_penalty = resources_->metadata.decoder.length_penalty;
    options.nbest = resources_->metadata.decoder.nbest;
    return std::make_unique<CtcWfstDecoder>(
        options, resources_->metadata.Resolve(resources_->metadata.decoder.graph));
  }
  if (type == DecoderType::kCtcPrefix) {
    CtcPrefixDecoderOptions options;
    options.blank_id = resources_->metadata.vocab.blank_id;
    return std::make_unique<CtcPrefixDecoder>(options);
  }
  GreedyCtcDecoderOptions options;
  options.blank_id = resources_->metadata.vocab.blank_id;
  return std::make_unique<GreedyCtcDecoder>(options);
}

}  // namespace wenet_sdk::internal
