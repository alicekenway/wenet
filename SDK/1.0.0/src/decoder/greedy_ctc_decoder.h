#ifndef WENET_SDK_SRC_DECODER_GREEDY_CTC_DECODER_H_
#define WENET_SDK_SRC_DECODER_GREEDY_CTC_DECODER_H_

#include <vector>

#include "decoder/decoder_interface.h"

namespace wenet_sdk::internal {

struct GreedyCtcDecoderOptions {
  int blank_id = 0;
};

std::vector<int> CollapseCtc(const std::vector<int>& frame_best_ids,
                             int blank_id);

class GreedyCtcDecoder final : public StreamingDecoder {
 public:
  explicit GreedyCtcDecoder(GreedyCtcDecoderOptions options);

  void Reset() override;
  void Advance(const std::vector<std::vector<float>>& log_probs) override;
  DecodeResult PartialResult() const override;
  DecodeResult Finalize() override;

 private:
  DecodeResult BuildResult() const;

  GreedyCtcDecoderOptions options_;
  std::vector<int> frame_best_ids_;
  std::vector<float> frame_best_logp_;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_DECODER_GREEDY_CTC_DECODER_H_
