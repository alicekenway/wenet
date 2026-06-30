#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_ASR_STREAM_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_ASR_STREAM_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "asr_sdk/config.h"
#include "asr_sdk/stream.h"
#include "flashlight_decoder/flashlight_decoder_resource.h"
#include "sherpa_onnx_wenet/zipformer2_ctc_onnx_backend.h"

namespace asr_sdk::internal::flashlight_decoder {

struct FlashlightAsrResources {
  std::shared_ptr<sherpa_onnx_wenet::Zipformer2CtcOnnxResource>
      acoustic_resource;
  FlashlightDecoderResourcePtr decoder_resource;
  EngineConfig config;
  std::string feature_type = "whisper";
};

class FlashlightAsrStream final : public AsrStream {
 public:
  explicit FlashlightAsrStream(std::shared_ptr<FlashlightAsrResources> shared);

  Status AcceptPcm16(const int16_t* samples, size_t num_samples,
                     int sample_rate) override;
  bool DecodeReady() const override;
  Status Decode() override;
  AsrResult GetResult() const override;
  AsrResult GetFinalResult() override;
  Status SetInputFinished() override;
  Status Reset() override;

 private:
  struct Impl;

  Status EnsureInitialized();
  Status ProcessQueuedFeatures();
  Status ForwardCurrentWindow(bool final_padding);
  Status EmitPartialIfAvailable();
  Status FinalizeIfReady();

  std::shared_ptr<FlashlightAsrResources> shared_;
  std::unique_ptr<Impl> impl_;
  bool input_finished_ = false;
  bool final_emitted_ = false;
  AsrResult last_result_;
  AsrResult final_result_;
};

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_ASR_STREAM_H_
