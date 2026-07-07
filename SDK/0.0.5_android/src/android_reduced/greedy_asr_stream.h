#ifndef ASR_SDK_SRC_ANDROID_REDUCED_GREEDY_ASR_STREAM_H_
#define ASR_SDK_SRC_ANDROID_REDUCED_GREEDY_ASR_STREAM_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "asr_sdk/config.h"
#include "asr_sdk/stream.h"
#include "package/model_package.h"
#include "sherpa_onnx_wenet/streaming_ctc_backend.h"
#include "sherpa_onnx_wenet/token_table.h"

namespace asr_sdk::internal::android_reduced {

struct GreedyAsrResources {
  std::shared_ptr<sherpa_onnx_wenet::StreamingCtcBackend> backend_template;
  std::shared_ptr<sherpa_onnx_wenet::TokenTable> tokens;
  EngineConfig config;
  std::string feature_type = "whisper";
};

StatusOr<std::shared_ptr<GreedyAsrResources>> CreateGreedyAsrResources(
    const EngineConfig& config, const ModelPackage& package);

class GreedyAsrStream final : public AsrStream {
 public:
  explicit GreedyAsrStream(std::shared_ptr<GreedyAsrResources> shared);

  Status AcceptPcm16(const int16_t* samples, size_t num_samples,
                     int sample_rate) override;
  bool DecodeReady() const override;
  Status Decode() override;
  AsrResult GetResult() const override;
  AsrResult GetFinalResult() override;
  Status SetInputFinished() override;
  Status Reset() override;

 private:
  Status DecodeAll();

  std::shared_ptr<GreedyAsrResources> shared_;
  std::vector<int16_t> samples_;
  bool input_finished_ = false;
  bool decoded_ = false;
  AsrResult result_;
};

}  // namespace asr_sdk::internal::android_reduced

#endif  // ASR_SDK_SRC_ANDROID_REDUCED_GREEDY_ASR_STREAM_H_
