#include "wenet_bridge/wenet_stream_adapter.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "decoder/asr_decoder.h"
#include "frontend/feature_pipeline.h"
#include "wenet_bridge/wenet_result_mapper.h"
#include "wenet_bridge/wenet_shared.h"

namespace asr_sdk::internal {
namespace {

void DeleteDecoder(void* decoder) {
  delete static_cast<wenet::AsrDecoder*>(decoder);
}

wenet::AsrDecoder* DecoderPtr(
    const std::unique_ptr<void, void (*)(void*)>& decoder) {
  return static_cast<wenet::AsrDecoder*>(decoder.get());
}

std::shared_ptr<wenet::FeaturePipeline> FeaturePipelinePtr(
    const std::shared_ptr<void>& feature_pipeline) {
  return std::static_pointer_cast<wenet::FeaturePipeline>(feature_pipeline);
}

AsrResult BuildResult(const std::vector<wenet::DecodeResult>& wenet_results,
                      bool final_result) {
  AsrResult result;
  result.is_final = final_result;
  for (const auto& item : wenet_results) {
    NBestResult nbest;
    nbest.text = item.sentence;
    nbest.score = item.score;
    result.nbest.push_back(nbest);
  }
  if (!result.nbest.empty()) {
    result.text = result.nbest.front().text;
    result.confidence = result.nbest.front().score;
  }
  if (final_result && !wenet_results.empty()) {
    for (const auto& piece : wenet_results.front().word_pieces) {
      TokenResult token;
      token.token = piece.word;
      token.start_ms = static_cast<float>(piece.start);
      token.end_ms = static_cast<float>(piece.end);
      result.tokens.push_back(std::move(token));
    }
  }
  return result;
}

}  // namespace

WenetStreamAdapter::WenetStreamAdapter(
    std::shared_ptr<wenet_types::Shared> shared,
                                       EngineConfig config)
    : shared_(std::move(shared)),
      config_(std::move(config)),
      decoder_(nullptr, DeleteDecoder),
      last_status_(Status::Ok()) {}

WenetStreamAdapter::~WenetStreamAdapter() {
  decoder_.reset();
}

Status WenetStreamAdapter::Init() {
  if (decoder_) {
    return Status::Ok();
  }
  if (!shared_) {
    SetStatus(Status::Internal("missing WeNet shared runtime resources"));
    return last_status_;
  }
  feature_pipeline_ =
      std::make_shared<wenet::FeaturePipeline>(*shared_->feature_config);
  auto typed_decoder = new wenet::AsrDecoder(
      FeaturePipelinePtr(feature_pipeline_), shared_->resource,
      *shared_->decode_options);
  decoder_.reset(typed_decoder);
  SetStatus(Status::Ok());
  return last_status_;
}

Status WenetStreamAdapter::AcceptPcm16(const int16_t* samples,
                                       size_t num_samples, int sample_rate) {
  if (!decoder_) {
    return SetStatus(Status::FailedPrecondition("stream is not initialized")),
           last_status_;
  }
  if (sample_rate != config_.sample_rate) {
    return SetStatus(Status::InvalidArgument(
               "sample_rate must be " + std::to_string(config_.sample_rate))),
           last_status_;
  }
  if (samples == nullptr && num_samples > 0) {
    return SetStatus(Status::InvalidArgument("samples is null")), last_status_;
  }
  if (input_finished_) {
    return SetStatus(Status::FailedPrecondition(
               "cannot accept audio after input is finished")),
           last_status_;
  }
  if (num_samples > 0) {
    pending_.insert(pending_.end(), samples, samples + num_samples);
  }
  SetStatus(Status::Ok());
  return last_status_;
}

bool WenetStreamAdapter::DecodeReady() const {
  return !pending_.empty() || (input_finished_ && !final_emitted_);
}

Status WenetStreamAdapter::Decode() {
  if (!decoder_) {
    return SetStatus(Status::FailedPrecondition("stream is not initialized")),
           last_status_;
  }
  if (!DecodeReady()) {
    SetStatus(Status::Ok());
    return last_status_;
  }

  auto feature_pipeline = FeaturePipelinePtr(feature_pipeline_);
  if (!pending_.empty()) {
    feature_pipeline->AcceptWaveform(pending_.data(),
                                     static_cast<int>(pending_.size()));
  }
  pending_.clear();
  if (input_finished_ && !feature_input_finished_sent_) {
    feature_pipeline->set_input_finished();
    feature_input_finished_sent_ = true;
  }

  auto* decoder = DecoderPtr(decoder_);
  while (true) {
    const wenet::DecodeState state = decoder->Decode(false);
    if (state == wenet::DecodeState::kWaitFeats) {
      last_result_ = BuildResult(decoder->result(), false);
      break;
    }
    if (state == wenet::DecodeState::kEndFeats) {
      decoder->Rescoring();
      last_result_ = BuildResult(decoder->result(), true);
      final_result_ = last_result_;
      final_emitted_ = true;
      break;
    }
    if (state == wenet::DecodeState::kEndpoint &&
        config_.enable_continuous_decoding) {
      decoder->Rescoring();
      last_result_ = BuildResult(decoder->result(), true);
      final_result_ = last_result_;
      decoder->ResetContinuousDecoding();
      break;
    }
    last_result_ = BuildResult(decoder->result(), false);
  }
  SetStatus(Status::Ok());
  return last_status_;
}

AsrResult WenetStreamAdapter::GetResult() const { return last_result_; }

AsrResult WenetStreamAdapter::GetFinalResult() {
  if (!final_emitted_) {
    input_finished_ = true;
    (void)Decode();
  }
  return final_result_;
}

Status WenetStreamAdapter::SetInputFinished() {
  input_finished_ = true;
  SetStatus(Status::Ok());
  return last_status_;
}

Status WenetStreamAdapter::Reset() {
  if (!decoder_) {
    return SetStatus(Status::FailedPrecondition("stream is not initialized")),
           last_status_;
  }
  DecoderPtr(decoder_)->Reset();
  pending_.clear();
  input_finished_ = false;
  feature_input_finished_sent_ = false;
  final_emitted_ = false;
  last_result_ = AsrResult();
  final_result_ = AsrResult();
  SetStatus(Status::Ok());
  return last_status_;
}

void WenetStreamAdapter::SetStatus(Status status) {
  last_status_ = std::move(status);
}

}  // namespace asr_sdk::internal
