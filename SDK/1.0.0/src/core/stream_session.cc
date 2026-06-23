#include "core/stream_session.h"

namespace wenet_sdk::internal {

StreamSession::StreamSession(std::shared_ptr<const EngineResources> resources)
    : recognizer_(std::move(resources)) {}

Status StreamSession::Init() {
  auto status = recognizer_.Init();
  SetStatus(status);
  return status;
}

void StreamSession::AcceptWaveform(int sample_rate, const float* samples,
                                   size_t n) {
  std::lock_guard<std::mutex> lock(mutex_);
  SetStatus(recognizer_.AcceptWaveform(sample_rate, samples, n));
}

bool StreamSession::DecodeReady() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return recognizer_.DecodeReady();
}

void StreamSession::Decode() {
  std::lock_guard<std::mutex> lock(mutex_);
  SetStatus(recognizer_.Decode());
}

AsrResult StreamSession::GetResult() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return recognizer_.GetResult();
}

AsrResult StreamSession::GetFinalResult() {
  std::lock_guard<std::mutex> lock(mutex_);
  return recognizer_.GetFinalResult();
}

void StreamSession::SetInputFinished() {
  std::lock_guard<std::mutex> lock(mutex_);
  recognizer_.SetInputFinished();
  SetStatus(Status::OK());
}

void StreamSession::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  recognizer_.Reset();
  SetStatus(Status::OK());
}

Status StreamSession::last_status() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return last_status_;
}

void StreamSession::SetStatus(Status status) const {
  last_status_ = std::move(status);
}

}  // namespace wenet_sdk::internal
