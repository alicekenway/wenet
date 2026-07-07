#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_CTC_STREAM_DECODER_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_CTC_STREAM_DECODER_H_

#include <memory>
#include <vector>

#include "asr_sdk/status.h"
#include "flashlight_decoder/decoded_hypothesis.h"
#include "flashlight_decoder/flashlight_decoder_resource.h"

namespace asr_sdk::internal::flashlight_decoder {

class FlashlightCtcStreamDecoder {
 public:
  explicit FlashlightCtcStreamDecoder(FlashlightDecoderResourcePtr resource);
  ~FlashlightCtcStreamDecoder();

  Status Start();
  Status DecodeChunk(const float* data, int frames, int vocab_size);
  StatusOr<DecodedHypothesis> PartialResult() const;
  StatusOr<std::vector<DecodedHypothesis>> Finalize();
  Status Reset();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_CTC_STREAM_DECODER_H_
