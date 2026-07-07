#ifndef ASR_SDK_SRC_SHERPA_ONNX_WENET_STREAMING_CTC_MODEL_INFO_H_
#define ASR_SDK_SRC_SHERPA_ONNX_WENET_STREAMING_CTC_MODEL_INFO_H_

namespace asr_sdk::internal::sherpa_onnx_wenet {

struct StreamingCtcModelInfo {
  int sample_rate = 16000;
  int feature_dim = 80;
  int input_window_frames = 0;
  int input_shift_frames = 0;
  int vocab_size = 0;
  int blank_id = 0;
  float output_frame_shift_ms = 0.0f;
  bool output_is_log_probs = true;
};

}  // namespace asr_sdk::internal::sherpa_onnx_wenet

#endif  // ASR_SDK_SRC_SHERPA_ONNX_WENET_STREAMING_CTC_MODEL_INFO_H_
