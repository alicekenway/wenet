#ifndef WENET_SDK_CONFIG_H_
#define WENET_SDK_CONFIG_H_

#include <string>

namespace wenet_sdk {

enum class DecoderType {
  kAuto,
  kGreedyCtc,
  kCtcPrefix,
  kCtcWfst,
};

struct EngineConfig {
  std::string model_dir;
  int num_threads = 1;
  bool enable_endpoint = true;
  bool enable_timestamps = false;
  DecoderType decoder_type = DecoderType::kAuto;

  int chunk_size = 0;
  int num_left_chunks = 0;
  float blank_skip_thresh = 0.0f;
  float waveform_scale = 0.0f;
};

}  // namespace wenet_sdk

#endif  // WENET_SDK_CONFIG_H_
