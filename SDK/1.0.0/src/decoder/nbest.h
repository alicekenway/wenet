#ifndef WENET_SDK_SRC_DECODER_NBEST_H_
#define WENET_SDK_SRC_DECODER_NBEST_H_

#include <vector>

namespace wenet_sdk::internal {

struct NBestPath {
  std::vector<int> token_ids;
  std::vector<int> word_ids;
  std::vector<int> frame_indexes;
  float acoustic_score = 0.0f;
  float lm_score = 0.0f;
  float total_score = 0.0f;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_DECODER_NBEST_H_
