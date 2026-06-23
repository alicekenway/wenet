#include <cassert>
#include <vector>

#include "decoder/greedy_ctc_decoder.h"

int main() {
  const int blank = 0;
  const std::vector<int> best_path = {0, 1, 1, 0, 2, 2, 0, 2, 3, 0};
  const std::vector<int> expected = {1, 2, 2, 3};
  assert(wenet_sdk::internal::CollapseCtc(best_path, blank) == expected);
  return 0;
}
