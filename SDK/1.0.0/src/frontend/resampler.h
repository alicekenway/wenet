#ifndef WENET_SDK_SRC_FRONTEND_RESAMPLER_H_
#define WENET_SDK_SRC_FRONTEND_RESAMPLER_H_

#include <vector>

namespace wenet_sdk::internal {

std::vector<float> LinearResample(const std::vector<float>& input,
                                  int input_rate, int output_rate);

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_FRONTEND_RESAMPLER_H_
