#include "frontend/resampler.h"

#include <algorithm>
#include <cmath>

namespace wenet_sdk::internal {

std::vector<float> LinearResample(const std::vector<float>& input,
                                  int input_rate, int output_rate) {
  if (input.empty() || input_rate <= 0 || output_rate <= 0 ||
      input_rate == output_rate) {
    return input;
  }
  const double ratio = static_cast<double>(output_rate) / input_rate;
  const size_t output_size =
      std::max<size_t>(1, static_cast<size_t>(std::round(input.size() * ratio)));
  std::vector<float> output(output_size, 0.0f);
  for (size_t i = 0; i < output_size; ++i) {
    const double src = i / ratio;
    const size_t lo = static_cast<size_t>(std::floor(src));
    const size_t hi = std::min(input.size() - 1, lo + 1);
    const double t = src - lo;
    output[i] = static_cast<float>((1.0 - t) * input[lo] + t * input[hi]);
  }
  return output;
}

}  // namespace wenet_sdk::internal
