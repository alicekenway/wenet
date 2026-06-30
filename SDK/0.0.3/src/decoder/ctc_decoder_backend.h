#ifndef ASR_SDK_SRC_DECODER_CTC_DECODER_BACKEND_H_
#define ASR_SDK_SRC_DECODER_CTC_DECODER_BACKEND_H_

#include <vector>

#include "asr_sdk/status.h"
#include "flashlight_decoder/decoded_hypothesis.h"

namespace asr_sdk::internal {

class CtcDecoderBackend {
 public:
  virtual ~CtcDecoderBackend() = default;

  virtual Status Start() = 0;
  virtual Status DecodeChunk(const float* data, int frames, int vocab_size) = 0;
  virtual StatusOr<flashlight_decoder::DecodedHypothesis> PartialResult()
      const = 0;
  virtual StatusOr<std::vector<flashlight_decoder::DecodedHypothesis>>
  Finalize() = 0;
  virtual Status Reset() = 0;
};

}  // namespace asr_sdk::internal

#endif  // ASR_SDK_SRC_DECODER_CTC_DECODER_BACKEND_H_
