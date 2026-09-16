#ifndef ASR_SDK_SRC_WENET_BRIDGE_WENET_STREAM_ADAPTER_H_
#define ASR_SDK_SRC_WENET_BRIDGE_WENET_STREAM_ADAPTER_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "asr_sdk/config.h"
#include "asr_sdk/result.h"
#include "asr_sdk/status.h"

namespace asr_sdk::internal {

namespace wenet_types {
struct Shared;
}  // namespace wenet_types

class WenetStreamAdapter {
 public:
  WenetStreamAdapter(std::shared_ptr<wenet_types::Shared> shared,
                     EngineConfig config);
  ~WenetStreamAdapter();

  Status Init();
  Status AcceptPcm16(const int16_t* samples, size_t num_samples,
                     int sample_rate);
  bool DecodeReady() const;
  Status Decode();
  AsrResult GetResult() const;
  AsrResult GetFinalResult();
  Status SetInputFinished();
  Status Reset();
  const Status& last_status() const { return last_status_; }

 private:
  void SetStatus(Status status);

  std::shared_ptr<wenet_types::Shared> shared_;
  EngineConfig config_;
  std::shared_ptr<void> feature_pipeline_;
  std::unique_ptr<void, void (*)(void*)> decoder_{nullptr, nullptr};
  std::vector<int16_t> pending_;
  bool input_finished_ = false;
  bool feature_input_finished_sent_ = false;
  bool final_emitted_ = false;
  AsrResult last_result_;
  AsrResult final_result_;
  Status last_status_;
};

}  // namespace asr_sdk::internal

#endif  // ASR_SDK_SRC_WENET_BRIDGE_WENET_STREAM_ADAPTER_H_
