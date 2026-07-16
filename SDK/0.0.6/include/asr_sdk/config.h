#ifndef ASR_SDK_CONFIG_H_
#define ASR_SDK_CONFIG_H_

#include <string>

namespace asr_sdk {

struct EngineConfig {
  std::string model_dir;
  int sample_rate = 16000;
  int num_threads = 1;
  int chunk_size = 16;
  int num_left_chunks = 16;
  int nbest = 1;
  bool enable_continuous_decoding = true;
  bool enable_timestamps = false;
  std::string language = "chs";
  bool debug = false;
};

}  // namespace asr_sdk

#endif  // ASR_SDK_CONFIG_H_
