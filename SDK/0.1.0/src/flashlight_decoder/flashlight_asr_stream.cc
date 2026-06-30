#include "flashlight_decoder/flashlight_asr_stream.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "flashlight_decoder/debug_trace.h"
#include "flashlight_decoder/flashlight_ctc_stream_decoder.h"
#include "frontend/feature_pipeline.h"
#include "sherpa_onnx_wenet/streaming_ctc_backend.h"
#include "sherpa_onnx_wenet/whisper_feature_extractor.h"

namespace asr_sdk::internal::flashlight_decoder {
namespace {

using sherpa_onnx_wenet::StreamingCtcBackend;

constexpr int kInitialPaddingMs = 300;
constexpr int kFinalPaddingMs = 800;

wenet::FeatureType ToWenetFeatureType(
    sherpa_onnx_wenet::ZipformerFeatureType type) {
  switch (type) {
    case sherpa_onnx_wenet::ZipformerFeatureType::kKaldi:
      return wenet::FeatureType::kKaldi;
    case sherpa_onnx_wenet::ZipformerFeatureType::kWhisper:
      return wenet::FeatureType::kWhisper;
  }
  throw std::runtime_error("unknown Zipformer feature type");
}

void FlattenFrames(const std::vector<std::vector<float>>& frames,
                   std::vector<float>* flat) {
  flat->clear();
  if (frames.empty()) {
    return;
  }
  flat->reserve(frames.size() * frames[0].size());
  for (const auto& frame : frames) {
    flat->insert(flat->end(), frame.begin(), frame.end());
  }
}

void AppendFrameOrZeros(const std::vector<std::vector<float>>& frames,
                        int frame_index, int dim, std::vector<float>* flat) {
  if (frame_index < static_cast<int>(frames.size())) {
    const auto& frame = frames[static_cast<size_t>(frame_index)];
    flat->insert(flat->end(), frame.begin(), frame.end());
  } else {
    flat->insert(flat->end(), static_cast<size_t>(dim), 0.0f);
  }
}

AsrResult BuildAsrResult(const std::vector<DecodedHypothesis>& hyps,
                         bool is_final, bool debug,
                         const std::vector<std::string>& logs,
                         const std::string& error) {
  AsrResult result;
  result.is_final = is_final;
  for (const DecodedHypothesis& hyp : hyps) {
    NBestResult nbest;
    nbest.text = JoinWords(hyp.mapped_words, " ");
    nbest.score = static_cast<float>(hyp.total_score);
    for (const DecodedWord& word : hyp.mapped_words) {
      TokenResult token;
      token.token = word.text;
      token.token_id = word.word_id;
      token.start_ms = word.start_frame * 40.0f;
      token.end_ms = word.end_frame * 40.0f;
      token.confidence = 0.0f;
      nbest.tokens.push_back(token);
    }
    result.nbest.push_back(std::move(nbest));
  }
  if (!result.nbest.empty()) {
    result.text = result.nbest.front().text;
    result.confidence = result.nbest.front().score;
    result.tokens = result.nbest.front().tokens;
  }
  if (debug) {
    result.raw_backend_json = BuildDebugJson(logs, error, hyps, is_final);
  }
  return result;
}

Status ExceptionStatus(const char* where, const std::exception& e) {
  return Status::Internal(std::string(where) + ": " + e.what());
}

void AcceptZeros(wenet::FeaturePipeline* pipeline, int sample_rate,
                 int duration_ms) {
  const int num_samples = sample_rate * duration_ms / 1000;
  std::vector<int16_t> zeros(static_cast<size_t>(num_samples), 0);
  pipeline->AcceptWaveform(zeros.data(), num_samples);
}

}  // namespace

struct FlashlightAsrStream::Impl {
  explicit Impl(std::shared_ptr<FlashlightAsrResources> shared_in)
      : shared(std::move(shared_in)),
        feature_config(std::make_unique<wenet::FeaturePipelineConfig>(
            80, shared->config.sample_rate,
            ToWenetFeatureType(sherpa_onnx_wenet::ParseZipformerFeatureType(
                shared->feature_type)))),
        feature_pipeline(
            std::make_unique<wenet::FeaturePipeline>(*feature_config)),
        backend(shared->backend_template->CloneStream()),
        decoder(std::make_unique<FlashlightCtcStreamDecoder>(
            shared->decoder_resource)) {}

  std::shared_ptr<FlashlightAsrResources> shared;
  std::unique_ptr<wenet::FeaturePipelineConfig> feature_config;
  std::unique_ptr<wenet::FeaturePipeline> feature_pipeline;
  std::unique_ptr<StreamingCtcBackend> backend;
  std::unique_ptr<FlashlightCtcStreamDecoder> decoder;
  std::vector<std::vector<float>> feature_buffer;
  std::vector<std::string> debug_logs;
  std::string debug_error;
  int pending_new_frames = 0;
  int decoded_chunks = 0;
  bool initial_padding_sent = false;
  bool final_padding_sent = false;
  bool feature_input_finished_sent = false;

  bool DebugEnabled() const { return shared && shared->config.debug; }

  void AddDebugLog(std::string line) {
    if (DebugEnabled()) {
      debug_logs.push_back(std::move(line));
    }
  }

  Status RecordStatus(Status status) {
    if (DebugEnabled() && !status.ok()) {
      debug_error = status.ToString();
      debug_logs.push_back("error " + debug_error);
    }
    return status;
  }
};

FlashlightAsrStream::FlashlightAsrStream(
    std::shared_ptr<FlashlightAsrResources> shared)
    : shared_(std::move(shared)) {}

Status FlashlightAsrStream::EnsureInitialized() {
  if (impl_) {
    return Status::Ok();
  }
  if (!shared_ || !shared_->backend_template || !shared_->decoder_resource) {
    return Status::Internal("missing Flashlight ASR resources");
  }
  try {
    impl_ = std::make_unique<Impl>(shared_);
    impl_->AddDebugLog("stream_init");
    AcceptZeros(impl_->feature_pipeline.get(), shared_->config.sample_rate,
                kInitialPaddingMs);
    impl_->initial_padding_sent = true;
    return Status::Ok();
  } catch (const std::exception& e) {
    return ExceptionStatus("Flashlight stream initialization failed", e);
  }
}

Status FlashlightAsrStream::AcceptPcm16(const int16_t* samples,
                                        size_t num_samples, int sample_rate) {
  if (!shared_) {
    return Status::Internal("missing Flashlight ASR resources");
  }
  if (sample_rate != shared_->config.sample_rate) {
    return Status::InvalidArgument("sample_rate must be " +
                                   std::to_string(shared_->config.sample_rate));
  }
  if (samples == nullptr && num_samples > 0) {
    return Status::InvalidArgument("samples is null");
  }
  if (input_finished_) {
    return Status::FailedPrecondition(
        "cannot accept audio after input is finished");
  }
  Status status = EnsureInitialized();
  if (!status.ok()) {
    return status;
  }
  try {
    impl_->AddDebugLog("accept_pcm samples=" + std::to_string(num_samples) +
                       " sample_rate=" + std::to_string(sample_rate));
    if (num_samples > 0) {
      impl_->feature_pipeline->AcceptWaveform(
          samples, static_cast<int>(num_samples));
    }
    return Status::Ok();
  } catch (const std::exception& e) {
    return ExceptionStatus("Flashlight AcceptPcm16 failed", e);
  }
}

bool FlashlightAsrStream::DecodeReady() const {
  if (final_emitted_) {
    return false;
  }
  if (impl_ && impl_->feature_pipeline &&
      impl_->feature_pipeline->NumQueuedFrames() > 0) {
    return true;
  }
  return input_finished_;
}

Status FlashlightAsrStream::ProcessQueuedFeatures() {
  if (!impl_) {
    return Status::FailedPrecondition("stream is not initialized");
  }
  try {
    while (impl_->feature_pipeline->NumQueuedFrames() > 0) {
      std::vector<float> frame;
      if (!impl_->feature_pipeline->ReadOne(&frame)) {
        break;
      }
      impl_->feature_buffer.push_back(std::move(frame));
      ++impl_->pending_new_frames;

      const int window = impl_->backend->Info().input_window_frames;
      while (static_cast<int>(impl_->feature_buffer.size()) >= window) {
        Status status = ForwardCurrentWindow(/*final_padding=*/false);
        if (!status.ok()) {
          return status;
        }
      }
    }
    return Status::Ok();
  } catch (const std::exception& e) {
    return ExceptionStatus("Flashlight feature processing failed", e);
  }
}

Status FlashlightAsrStream::ForwardCurrentWindow(bool final_padding) {
  if (!impl_) {
    return Status::FailedPrecondition("stream is not initialized");
  }
  const auto& info = impl_->backend->Info();
  const int window = info.input_window_frames;
  const int shift = info.input_shift_frames;
  const int dim = info.feature_dim;
  if (window <= shift || shift <= 0 || dim <= 0) {
    return Status::Internal("invalid CTC chunk geometry");
  }

  try {
    impl_->AddDebugLog("forward_window final_padding=" +
                       std::string(final_padding ? "true" : "false") +
                       " buffered_frames=" +
                       std::to_string(impl_->feature_buffer.size()));
    std::vector<float> chunk;
    chunk.reserve(static_cast<size_t>(window * dim));
    for (int frame = 0; frame < window; ++frame) {
      AppendFrameOrZeros(impl_->feature_buffer, frame, dim, &chunk);
    }

    std::vector<std::vector<float>> log_probs;
    impl_->backend->Forward(chunk.data(), window, &log_probs);
    std::vector<float> flat;
    FlattenFrames(log_probs, &flat);
    Status status = impl_->decoder->DecodeChunk(
        flat.data(), static_cast<int>(log_probs.size()), info.vocab_size);
    if (!status.ok()) {
      return impl_->RecordStatus(status);
    }
    impl_->AddDebugLog("decode_chunk frames=" +
                       std::to_string(log_probs.size()) + " vocab=" +
                       std::to_string(info.vocab_size));

    if (final_padding) {
      impl_->feature_buffer.clear();
      impl_->pending_new_frames = 0;
    } else {
      const int consumed_new =
          impl_->decoded_chunks == 0 ? window : shift;
      impl_->pending_new_frames =
          std::max(0, impl_->pending_new_frames - consumed_new);
      const int erase_count =
          std::min(shift, static_cast<int>(impl_->feature_buffer.size()));
      impl_->feature_buffer.erase(
          impl_->feature_buffer.begin(),
          impl_->feature_buffer.begin() + erase_count);
    }
    ++impl_->decoded_chunks;
    return Status::Ok();
  } catch (const std::exception& e) {
    return ExceptionStatus("Flashlight CTC decode chunk failed", e);
  }
}

Status FlashlightAsrStream::EmitPartialIfAvailable() {
  if (!impl_ || impl_->decoded_chunks == 0) {
    return Status::Ok();
  }
  auto partial_or = impl_->decoder->PartialResult();
  if (!partial_or.ok()) {
    return impl_->RecordStatus(partial_or.status());
  }
  std::vector<DecodedHypothesis> hyps;
  hyps.push_back(std::move(partial_or).value());
  impl_->AddDebugLog("partial_result text=" +
                     JoinWords(hyps.front().mapped_words, " "));
  last_result_ = BuildAsrResult(hyps, false, impl_->DebugEnabled(),
                                impl_->debug_logs, impl_->debug_error);
  return Status::Ok();
}

Status FlashlightAsrStream::FinalizeIfReady() {
  if (!input_finished_ || final_emitted_) {
    return Status::Ok();
  }
  if (!impl_) {
    return Status::FailedPrecondition("stream is not initialized");
  }
  if (impl_->feature_pipeline->NumQueuedFrames() > 0) {
    return Status::Ok();
  }
  if (impl_->pending_new_frames > 0 || impl_->decoded_chunks == 0) {
    Status status = ForwardCurrentWindow(/*final_padding=*/true);
    if (!status.ok()) {
      return impl_->RecordStatus(status);
    }
  }
  impl_->AddDebugLog("finalize_start");
  auto hyps_or = impl_->decoder->Finalize();
  if (!hyps_or.ok()) {
    return impl_->RecordStatus(hyps_or.status());
  }
  const auto& hyps = hyps_or.value();
  impl_->AddDebugLog("rescore_nbest count=" + std::to_string(hyps.size()));
  impl_->AddDebugLog("finalize_done nbest=" + std::to_string(hyps.size()));
  final_result_ = BuildAsrResult(hyps, true, impl_->DebugEnabled(),
                                 impl_->debug_logs, impl_->debug_error);
  last_result_ = final_result_;
  final_emitted_ = true;
  return Status::Ok();
}

Status FlashlightAsrStream::Decode() {
  if (!DecodeReady()) {
    return Status::Ok();
  }
  Status status = EnsureInitialized();
  if (!status.ok()) {
    return status;
  }
  status = ProcessQueuedFeatures();
  if (!status.ok()) {
    return status;
  }
  status = FinalizeIfReady();
  if (!status.ok() || final_emitted_) {
    return status;
  }
  return EmitPartialIfAvailable();
}

AsrResult FlashlightAsrStream::GetResult() const { return last_result_; }

AsrResult FlashlightAsrStream::GetFinalResult() {
  if (!final_emitted_) {
    Status status = SetInputFinished();
    while (status.ok() && DecodeReady()) {
      status = Decode();
    }
  }
  return final_result_;
}

Status FlashlightAsrStream::SetInputFinished() {
  if (input_finished_) {
    return Status::Ok();
  }
  input_finished_ = true;
  Status status = EnsureInitialized();
  if (!status.ok()) {
    return status;
  }
  try {
    if (!impl_->final_padding_sent) {
      AcceptZeros(impl_->feature_pipeline.get(), shared_->config.sample_rate,
                  kFinalPaddingMs);
      impl_->final_padding_sent = true;
    }
    if (!impl_->feature_input_finished_sent) {
      impl_->feature_pipeline->set_input_finished();
      impl_->feature_input_finished_sent = true;
    }
    return Status::Ok();
  } catch (const std::exception& e) {
    return ExceptionStatus("Flashlight SetInputFinished failed", e);
  }
}

Status FlashlightAsrStream::Reset() {
  impl_.reset();
  input_finished_ = false;
  final_emitted_ = false;
  last_result_ = AsrResult();
  final_result_ = AsrResult();
  return Status::Ok();
}

}  // namespace asr_sdk::internal::flashlight_decoder
