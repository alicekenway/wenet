#ifndef WENET_SDK_SRC_IO_WAV_WRITER_H_
#define WENET_SDK_SRC_IO_WAV_WRITER_H_

#include <filesystem>
#include <vector>

#include "utils/status.h"

namespace wenet_sdk::internal {

Status WriteWavFile(const std::filesystem::path& path, int sample_rate,
                    const std::vector<float>& samples);

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_IO_WAV_WRITER_H_
