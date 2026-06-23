#ifndef WENET_SDK_SRC_CORE_STREAM_SESSION_H_
#define WENET_SDK_SRC_CORE_STREAM_SESSION_H_

#include <memory>
#include <mutex>

#include "core/engine_resources.h"
#include "core/recognizer.h"
#include "utils/status.h"
#include "wenet_sdk/stream.h"

namespace wenet_sdk::internal {

class StreamSession final : public Stream {
 public:
  explicit StreamSession(std::shared_ptr<const EngineResources> resources);

  Status Init();

  void AcceptWaveform(int sample_rate, const float* samples, size_t n) override;
  bool DecodeReady() const override;
  void Decode() override;
  AsrResult GetResult() const override;
  AsrResult GetFinalResult() override;
  void SetInputFinished() override;
  void Reset() override;

  Status last_status() const;

 private:
  void SetStatus(Status status) const;

  mutable std::mutex mutex_;
  mutable Status last_status_;
  Recognizer recognizer_;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_CORE_STREAM_SESSION_H_
