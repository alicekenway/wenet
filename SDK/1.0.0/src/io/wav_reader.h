#ifndef WENET_SDK_SRC_IO_WAV_READER_H_
#define WENET_SDK_SRC_IO_WAV_READER_H_

#include <filesystem>
#include <vector>

#include "utils/status.h"

namespace wenet_sdk::internal {

struct WavData {
  int sample_rate = 0;
  int num_channels = 0;
  std::vector<float> samples;
};

Status ReadWavFile(const std::filesystem::path& path, WavData* data);

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_IO_WAV_READER_H_
