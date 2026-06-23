#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <vector>

#include "io/wav_reader.h"
#include "io/wav_writer.h"
#include "wenet_sdk/asr_engine.h"

namespace {

std::vector<float> MakeTone(int sample_rate, int num_samples) {
  std::vector<float> samples(static_cast<size_t>(num_samples));
  for (int i = 0; i < num_samples; ++i) {
    samples[static_cast<size_t>(i)] =
        0.2f * std::sin(2.0 * 3.14159265358979323846 * 440.0 *
                        static_cast<double>(i) / sample_rate);
  }
  return samples;
}

}  // namespace

int main() {
  const int sample_rate = 8000;
  const auto wav_path =
      std::filesystem::temp_directory_path() / "wenet_sdk_decode_wav.wav";
  const auto samples = MakeTone(sample_rate, sample_rate / 2);
  auto status =
      wenet_sdk::internal::WriteWavFile(wav_path, sample_rate, samples);
  assert(status.ok());

  wenet_sdk::internal::WavData wav;
  status = wenet_sdk::internal::ReadWavFile(wav_path, &wav);
  assert(status.ok());
  assert(wav.sample_rate == sample_rate);
  assert(!wav.samples.empty());

  wenet_sdk::EngineConfig config;
  config.model_dir = "model_example";
  config.enable_timestamps = true;
  auto engine = wenet_sdk::AsrEngine::Create(config);
  assert(engine);
  auto stream = engine->CreateStream();
  assert(stream);

  const size_t chunk = static_cast<size_t>(sample_rate / 10);
  for (size_t offset = 0; offset < wav.samples.size(); offset += chunk) {
    const size_t n = std::min(chunk, wav.samples.size() - offset);
    stream->AcceptWaveform(wav.sample_rate, wav.samples.data() + offset, n);
    while (stream->DecodeReady()) {
      stream->Decode();
    }
  }
  stream->SetInputFinished();
  while (stream->DecodeReady()) {
    stream->Decode();
  }
  const auto result = stream->GetFinalResult();
  assert(result.is_final);
  assert(!result.text.empty());
#ifdef WENETSDK_ENABLE_ONNX
  assert(result.text == "hello world sdk");
#endif
  std::filesystem::remove(wav_path);
  return 0;
}
