#include <cassert>
#include <cmath>
#include <vector>

#include "decoder/ctc_prefix_decoder.h"

int main() {
  wenet_sdk::internal::CtcPrefixDecoder decoder({0, 3, 3});
  const std::vector<std::vector<float>> logp = {
      {std::log(0.90f), std::log(0.05f), std::log(0.05f)},
      {std::log(0.05f), std::log(0.90f), std::log(0.05f)},
      {std::log(0.90f), std::log(0.05f), std::log(0.05f)},
      {std::log(0.05f), std::log(0.05f), std::log(0.90f)}};
  decoder.Advance(logp);
  const auto result = decoder.Finalize();
  const std::vector<int> expected = {1, 2};
  assert(result.token_ids == expected);
  assert(result.frame_indexes.size() == expected.size());
  return 0;
}
