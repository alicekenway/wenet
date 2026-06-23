#include "io/wav_writer.h"

#include <algorithm>
#include <cstdint>
#include <fstream>

namespace wenet_sdk::internal {
namespace {

void WriteU16(std::ofstream* out, uint16_t value) {
  const char b[2] = {static_cast<char>(value & 0xff),
                     static_cast<char>((value >> 8) & 0xff)};
  out->write(b, 2);
}

void WriteU32(std::ofstream* out, uint32_t value) {
  const char b[4] = {static_cast<char>(value & 0xff),
                     static_cast<char>((value >> 8) & 0xff),
                     static_cast<char>((value >> 16) & 0xff),
                     static_cast<char>((value >> 24) & 0xff)};
  out->write(b, 4);
}

}  // namespace

Status WriteWavFile(const std::filesystem::path& path, int sample_rate,
                    const std::vector<float>& samples) {
  std::ofstream out(path, std::ios::binary);
  if (!out) {
    return Status::Internal("failed to write wav: " + path.string());
  }
  const uint32_t data_bytes =
      static_cast<uint32_t>(samples.size() * sizeof(int16_t));
  out.write("RIFF", 4);
  WriteU32(&out, 36 + data_bytes);
  out.write("WAVEfmt ", 8);
  WriteU32(&out, 16);
  WriteU16(&out, 1);
  WriteU16(&out, 1);
  WriteU32(&out, static_cast<uint32_t>(sample_rate));
  WriteU32(&out, static_cast<uint32_t>(sample_rate * sizeof(int16_t)));
  WriteU16(&out, sizeof(int16_t));
  WriteU16(&out, 16);
  out.write("data", 4);
  WriteU32(&out, data_bytes);
  for (float sample : samples) {
    const float clipped = std::max(-1.0f, std::min(1.0f, sample));
    const int16_t pcm = static_cast<int16_t>(clipped * 32767.0f);
    WriteU16(&out, static_cast<uint16_t>(pcm));
  }
  return Status::OK();
}

}  // namespace wenet_sdk::internal
