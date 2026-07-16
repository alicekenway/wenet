#include "audio/wav_reader.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>

namespace asr_sdk::internal {
namespace {

uint32_t ReadU32(std::ifstream* in) {
  unsigned char b[4] = {0, 0, 0, 0};
  in->read(reinterpret_cast<char*>(b), 4);
  return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
         (static_cast<uint32_t>(b[2]) << 16) |
         (static_cast<uint32_t>(b[3]) << 24);
}

uint16_t ReadU16(std::ifstream* in) {
  unsigned char b[2] = {0, 0};
  in->read(reinterpret_cast<char*>(b), 2);
  return static_cast<uint16_t>(b[0]) | (static_cast<uint16_t>(b[1]) << 8);
}

std::string ReadTag(std::ifstream* in) {
  char tag[4] = {0, 0, 0, 0};
  in->read(tag, 4);
  return std::string(tag, 4);
}

}  // namespace

Status ReadWavFile(const std::filesystem::path& path, WavData* data) {
  if (data == nullptr) {
    return Status::InvalidArgument("wav output is null");
  }
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    return Status::NotFound("failed to open wav: " + path.string());
  }
  if (ReadTag(&in) != "RIFF") {
    return Status::InvalidArgument("wav is missing RIFF header: " +
                                   path.string());
  }
  (void)ReadU32(&in);
  if (ReadTag(&in) != "WAVE") {
    return Status::InvalidArgument("wav is missing WAVE header: " +
                                   path.string());
  }

  uint16_t audio_format = 0;
  uint16_t channels = 0;
  uint32_t sample_rate = 0;
  uint16_t bits_per_sample = 0;
  std::vector<char> pcm_bytes;

  while (in && (!audio_format || pcm_bytes.empty())) {
    const std::string tag = ReadTag(&in);
    if (!in) {
      break;
    }
    const uint32_t size = ReadU32(&in);
    if (tag == "fmt ") {
      audio_format = ReadU16(&in);
      channels = ReadU16(&in);
      sample_rate = ReadU32(&in);
      (void)ReadU32(&in);
      (void)ReadU16(&in);
      bits_per_sample = ReadU16(&in);
      if (size > 16) {
        in.seekg(static_cast<std::streamoff>(size - 16), std::ios::cur);
      }
    } else if (tag == "data") {
      pcm_bytes.resize(size);
      in.read(pcm_bytes.data(), static_cast<std::streamsize>(size));
    } else {
      in.seekg(static_cast<std::streamoff>(size), std::ios::cur);
    }
    if (size % 2 == 1) {
      in.seekg(1, std::ios::cur);
    }
  }

  const bool pcm16 = audio_format == 1 && bits_per_sample == 16;
  const bool float32 = audio_format == 3 && bits_per_sample == 32;
  if ((!pcm16 && !float32) || channels == 0) {
    return Status::InvalidArgument(
        "only 16-bit PCM or 32-bit IEEE float WAV files are supported by this tool");
  }
  data->sample_rate = static_cast<int>(sample_rate);
  data->num_channels = static_cast<int>(channels);
  data->samples.clear();

  const size_t bytes_per_sample = pcm16 ? sizeof(int16_t) : sizeof(float);
  if (pcm_bytes.size() % (bytes_per_sample * channels) != 0) {
    return Status::InvalidArgument("wav data chunk is not sample aligned: " +
                                   path.string());
  }
  const size_t total_samples = pcm_bytes.size() / bytes_per_sample;
  data->samples.reserve(total_samples / channels);
  for (size_t i = 0; i + channels <= total_samples; i += channels) {
    double mix = 0.0;
    for (uint16_t c = 0; c < channels; ++c) {
      const size_t byte = (i + c) * bytes_per_sample;
      if (pcm16) {
        const auto lo = static_cast<unsigned char>(pcm_bytes[byte]);
        const auto hi = static_cast<unsigned char>(pcm_bytes[byte + 1]);
        const int16_t sample =
            static_cast<int16_t>(lo | static_cast<uint16_t>(hi << 8));
        mix += sample;
      } else {
        uint32_t bits = static_cast<unsigned char>(pcm_bytes[byte]) |
                        (static_cast<uint32_t>(
                             static_cast<unsigned char>(pcm_bytes[byte + 1]))
                         << 8) |
                        (static_cast<uint32_t>(
                             static_cast<unsigned char>(pcm_bytes[byte + 2]))
                         << 16) |
                        (static_cast<uint32_t>(
                             static_cast<unsigned char>(pcm_bytes[byte + 3]))
                         << 24);
        float sample = 0.0f;
        std::memcpy(&sample, &bits, sizeof(sample));
        if (std::isfinite(sample)) {
          mix += std::clamp(static_cast<double>(sample), -1.0, 1.0) *
                 32767.0;
        }
      }
    }
    const double mono = mix / channels;
    const double clipped = std::clamp(
        mono, static_cast<double>(std::numeric_limits<int16_t>::min()),
        static_cast<double>(std::numeric_limits<int16_t>::max()));
    data->samples.push_back(static_cast<int16_t>(std::lround(clipped)));
  }
  return Status::Ok();
}

}  // namespace asr_sdk::internal
