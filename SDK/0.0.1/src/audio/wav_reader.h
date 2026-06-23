#ifndef ASR_SDK_SRC_AUDIO_WAV_READER_H_
#define ASR_SDK_SRC_AUDIO_WAV_READER_H_

#include <cstdint>
#include <filesystem>
#include <vector>

#include "asr_sdk/status.h"

namespace asr_sdk::internal {

struct WavData {
  int sample_rate = 0;
  int num_channels = 0;
  std::vector<int16_t> samples;
};

Status ReadWavFile(const std::filesystem::path& path, WavData* data);

}  // namespace asr_sdk::internal

#endif  // ASR_SDK_SRC_AUDIO_WAV_READER_H_
