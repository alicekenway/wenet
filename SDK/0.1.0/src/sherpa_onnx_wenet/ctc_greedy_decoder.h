#ifndef ASR_SDK_SRC_SHERPA_ONNX_WENET_CTC_GREEDY_DECODER_H_
#define ASR_SDK_SRC_SHERPA_ONNX_WENET_CTC_GREEDY_DECODER_H_

#include <vector>

namespace asr_sdk::internal::sherpa_onnx_wenet {

std::vector<int> CtcGreedyDecode(
    const std::vector<std::vector<float>>& log_probs, int blank_id);

}  // namespace asr_sdk::internal::sherpa_onnx_wenet

#endif  // ASR_SDK_SRC_SHERPA_ONNX_WENET_CTC_GREEDY_DECODER_H_
