#ifndef WENET_SDK_SRC_DECODER_DECODER_INTERFACE_H_
#define WENET_SDK_SRC_DECODER_DECODER_INTERFACE_H_

#include <vector>

#include "decoder/nbest.h"

namespace wenet_sdk::internal {

struct DecodeResult {
  std::vector<int> token_ids;
  std::vector<int> word_ids;
  std::vector<int> frame_indexes;
  float confidence = 0.0f;
  std::vector<NBestPath> nbest;
};

class StreamingDecoder {
 public:
  virtual ~StreamingDecoder() = default;

  virtual void Reset() = 0;
  virtual void Advance(const std::vector<std::vector<float>>& log_probs) = 0;
  virtual DecodeResult PartialResult() const = 0;
  virtual DecodeResult Finalize() = 0;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_DECODER_DECODER_INTERFACE_H_
