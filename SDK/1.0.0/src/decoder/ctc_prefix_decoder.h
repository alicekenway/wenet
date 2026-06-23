#ifndef WENET_SDK_SRC_DECODER_CTC_PREFIX_DECODER_H_
#define WENET_SDK_SRC_DECODER_CTC_PREFIX_DECODER_H_

#include <cstddef>
#include <unordered_map>
#include <limits>

#include "decoder/decoder_interface.h"

namespace wenet_sdk::internal {

struct CtcPrefixDecoderOptions {
  int blank_id = 0;
  int first_beam_size = 10;
  int second_beam_size = 10;
};

class CtcPrefixDecoder final : public StreamingDecoder {
 public:
  explicit CtcPrefixDecoder(CtcPrefixDecoderOptions options);

  void Reset() override;
  void Advance(const std::vector<std::vector<float>>& log_probs) override;
  DecodeResult PartialResult() const override;
  DecodeResult Finalize() override;

 private:
  struct PrefixScore {
    float blank = -std::numeric_limits<float>::max();
    float non_blank = -std::numeric_limits<float>::max();
    float viterbi_blank = -std::numeric_limits<float>::max();
    float viterbi_non_blank = -std::numeric_limits<float>::max();
    float current_token_logp = -std::numeric_limits<float>::max();
    std::vector<int> times_blank;
    std::vector<int> times_non_blank;

    float Score() const;
    float ViterbiScore() const;
    const std::vector<int>& Times() const;
  };

  struct PrefixHash {
    size_t operator()(const std::vector<int>& prefix) const {
      size_t hash = 0;
      for (int id : prefix) {
        hash = static_cast<size_t>(id) + 31 * hash;
      }
      return hash;
    }
  };

  DecodeResult BuildResult() const;
  void UpdateHypotheses(
      const std::vector<std::pair<std::vector<int>, PrefixScore>>& hyps);

  CtcPrefixDecoderOptions options_;
  int abs_time_step_ = 0;
  std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> cur_hyps_;
  std::vector<std::vector<int>> hypotheses_;
  std::vector<float> likelihood_;
  std::vector<std::vector<int>> times_;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_DECODER_CTC_PREFIX_DECODER_H_
