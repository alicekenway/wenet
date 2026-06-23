#include <cassert>
#include <cmath>
#include <vector>

#include "decoder/blank_skipper.h"

int main() {
  wenet_sdk::internal::BlankSkipper skipper({0, 0.9f, true});
  const std::vector<std::vector<float>> logp = {
      {std::log(0.99f), std::log(0.01f)},
      {std::log(0.20f), std::log(0.80f)}};
  const auto kept = skipper.Filter(logp);
  assert(!kept.empty());
  assert(skipper.frames_seen() == 2);
  assert(skipper.frames_skipped() >= 0);
  return 0;
}
