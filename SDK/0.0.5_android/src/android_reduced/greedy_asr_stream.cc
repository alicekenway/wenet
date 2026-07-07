#include "android_reduced/greedy_asr_stream.h"

#include <algorithm>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include "sherpa_onnx_wenet/ctc_greedy_decoder.h"
#include "sherpa_onnx_wenet/ctc_onnx_backend_factory.h"
#include "sherpa_onnx_wenet/whisper_feature_extractor.h"

namespace asr_sdk::internal::android_reduced {
namespace {

void AppendFrameOrZeros(const std::vector<std::vector<float>>& frames,
                        int frame_index, int dim, std::vector<float>* flat) {
  if (frame_index < static_cast<int>(frames.size())) {
    const auto& frame = frames[static_cast<size_t>(frame_index)];
    flat->insert(flat->end(), frame.begin(), frame.end());
  } else {
    flat->insert(flat->end(), static_cast<size_t>(dim), 0.0f);
  }
}

Status ExceptionStatus(const char* where, const std::exception& e) {
  return Status::Internal(std::string(where) + ": " + e.what());
}

}  // namespace

StatusOr<std::shared_ptr<GreedyAsrResources>> CreateGreedyAsrResources(
    const EngineConfig& config, const ModelPackage& package) {
  try {
    auto tokens =
        std::make_shared<sherpa_onnx_wenet::TokenTable>(package.tokens_txt);
    auto backend = sherpa_onnx_wenet::CreateStreamingCtcBackend(
        package.sherpa_ctc_onnx.string(), config.num_threads,
        tokens->BlankId());
    if (backend->Info().vocab_size != tokens->ModelVocabSize()) {
      return Status::InvalidArgument(
          "ONNX vocab size does not match tokens.txt model vocab size");
    }

    EngineConfig resolved = config;
    resolved.model_dir = package.root.string();
    resolved.sample_rate = package.sample_rate;
    resolved.nbest = package.nbest;
    resolved.debug = config.debug || package.debug;

    auto shared = std::make_shared<GreedyAsrResources>();
    shared->backend_template =
        std::shared_ptr<sherpa_onnx_wenet::StreamingCtcBackend>(
            std::move(backend));
    shared->tokens = std::move(tokens);
    shared->config = std::move(resolved);
    shared->feature_type = package.feature_type;
    return shared;
  } catch (const std::exception& e) {
    return ExceptionStatus("failed to load CTC greedy package", e);
  }
}

GreedyAsrStream::GreedyAsrStream(std::shared_ptr<GreedyAsrResources> shared)
    : shared_(std::move(shared)) {}

Status GreedyAsrStream::AcceptPcm16(const int16_t* samples, size_t num_samples,
                                    int sample_rate) {
  if (!shared_) {
    return Status::Internal("missing CTC greedy resources");
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
  if (num_samples > 0) {
    samples_.insert(samples_.end(), samples, samples + num_samples);
  }
  return Status::Ok();
}

bool GreedyAsrStream::DecodeReady() const {
  return input_finished_ && !decoded_;
}

Status GreedyAsrStream::Decode() {
  if (!DecodeReady()) {
    return Status::Ok();
  }
  return DecodeAll();
}

Status GreedyAsrStream::DecodeAll() {
  if (!shared_ || !shared_->backend_template || !shared_->tokens) {
    return Status::Internal("missing CTC greedy resources");
  }
  try {
    sherpa_onnx_wenet::WhisperFeatureOptions feature_options;
    feature_options.feature_type =
        sherpa_onnx_wenet::ParseZipformerFeatureType(shared_->feature_type);
    feature_options.sample_rate = shared_->config.sample_rate;

    std::vector<std::vector<float>> features;
    sherpa_onnx_wenet::WhisperFeatureExtractor(feature_options)
        .ExtractPcm16(samples_.data(), samples_.size(), &features);

    auto backend = shared_->backend_template->CloneStream();
    const auto& info = backend->Info();
    if (info.input_window_frames <= 0 || info.input_shift_frames <= 0 ||
        info.feature_dim <= 0 || info.vocab_size <= 0) {
      return Status::Internal("invalid CTC model geometry");
    }

    std::vector<std::vector<float>> all_log_probs;
    int start = 0;
    bool forwarded = false;
    while (start + info.input_window_frames <=
           static_cast<int>(features.size())) {
      std::vector<float> chunk;
      chunk.reserve(static_cast<size_t>(info.input_window_frames *
                                        info.feature_dim));
      for (int frame = 0; frame < info.input_window_frames; ++frame) {
        AppendFrameOrZeros(features, start + frame, info.feature_dim, &chunk);
      }
      std::vector<std::vector<float>> log_probs;
      backend->Forward(chunk.data(), info.input_window_frames, &log_probs);
      all_log_probs.insert(all_log_probs.end(), log_probs.begin(),
                           log_probs.end());
      start += info.input_shift_frames;
      forwarded = true;
    }

    if (!forwarded || start < static_cast<int>(features.size())) {
      std::vector<float> chunk;
      chunk.reserve(static_cast<size_t>(info.input_window_frames *
                                        info.feature_dim));
      for (int frame = 0; frame < info.input_window_frames; ++frame) {
        AppendFrameOrZeros(features, start + frame, info.feature_dim, &chunk);
      }
      std::vector<std::vector<float>> log_probs;
      backend->Forward(chunk.data(), info.input_window_frames, &log_probs);
      all_log_probs.insert(all_log_probs.end(), log_probs.begin(),
                           log_probs.end());
    }

    const std::vector<int> ids =
        sherpa_onnx_wenet::CtcGreedyDecode(all_log_probs,
                                           shared_->tokens->BlankId());
    result_ = AsrResult();
    result_.is_final = true;
    result_.text = shared_->tokens->DecodeIds(ids);
    NBestResult nbest;
    nbest.text = result_.text;
    nbest.score = 0.0f;
    const float frame_shift =
        info.output_frame_shift_ms > 0.0f ? info.output_frame_shift_ms : 40.0f;
    for (size_t i = 0; i < ids.size(); ++i) {
      TokenResult token;
      token.token_id = ids[i];
      token.token = shared_->tokens->Token(ids[i]);
      token.start_ms = static_cast<float>(i) * frame_shift;
      token.end_ms = static_cast<float>(i + 1) * frame_shift;
      token.confidence = 0.0f;
      result_.tokens.push_back(token);
      nbest.tokens.push_back(token);
    }
    result_.nbest.push_back(std::move(nbest));
    decoded_ = true;
    return Status::Ok();
  } catch (const std::exception& e) {
    return ExceptionStatus("CTC greedy decode failed", e);
  }
}

AsrResult GreedyAsrStream::GetResult() const { return result_; }

AsrResult GreedyAsrStream::GetFinalResult() {
  if (!decoded_) {
    Status status = SetInputFinished();
    while (status.ok() && DecodeReady()) {
      status = Decode();
    }
  }
  return result_;
}

Status GreedyAsrStream::SetInputFinished() {
  input_finished_ = true;
  return Status::Ok();
}

Status GreedyAsrStream::Reset() {
  samples_.clear();
  input_finished_ = false;
  decoded_ = false;
  result_ = AsrResult();
  return Status::Ok();
}

}  // namespace asr_sdk::internal::android_reduced
