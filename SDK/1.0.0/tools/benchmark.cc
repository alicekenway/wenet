#include <algorithm>
#include <iostream>
#include <string>

#include "io/wav_reader.h"
#include "utils/timer.h"
#include "wenet_sdk/asr_engine.h"

namespace {

std::string ArgValue(int argc, char** argv, const std::string& name,
                     const std::string& fallback = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) {
      return argv[i + 1];
    }
  }
  return fallback;
}

int ArgInt(int argc, char** argv, const std::string& name, int fallback) {
  const std::string value = ArgValue(argc, argv, name);
  if (value.empty()) {
    return fallback;
  }
  try {
    return std::stoi(value);
  } catch (...) {
    return fallback;
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model_dir = ArgValue(argc, argv, "--model_dir");
  const std::string wav = ArgValue(argc, argv, "--wav");
  const int chunk_ms = ArgInt(argc, argv, "--chunk_ms", 100);
  if (model_dir.empty() || wav.empty()) {
    std::cerr << "usage: benchmark --model_dir MODEL_DIR --wav WAV "
                 "[--chunk_ms 100]\n";
    return 2;
  }

  wenet_sdk::internal::WavData wav_data;
  auto status = wenet_sdk::internal::ReadWavFile(wav, &wav_data);
  if (!status.ok()) {
    std::cerr << status.message() << "\n";
    return 1;
  }

  wenet_sdk::EngineConfig config;
  config.model_dir = model_dir;
  auto engine = wenet_sdk::AsrEngine::Create(config);
  if (!engine) {
    std::cerr << "failed to create engine\n";
    return 1;
  }
  auto stream = engine->CreateStream();
  if (!stream) {
    std::cerr << "failed to create stream\n";
    return 1;
  }

  wenet_sdk::internal::Timer timer;
  double first_partial_latency_ms = -1.0;
  int partial_updates = 0;
  int chunks = 0;
  const size_t chunk_samples =
      static_cast<size_t>(std::max(1, wav_data.sample_rate * chunk_ms / 1000));
  for (size_t offset = 0; offset < wav_data.samples.size();
       offset += chunk_samples) {
    const size_t n =
        std::min(chunk_samples, wav_data.samples.size() - offset);
    stream->AcceptWaveform(wav_data.sample_rate, wav_data.samples.data() + offset,
                           n);
    while (stream->DecodeReady()) {
      stream->Decode();
      ++chunks;
      const auto partial = stream->GetResult();
      if (!partial.text.empty()) {
        ++partial_updates;
        if (first_partial_latency_ms < 0.0) {
          first_partial_latency_ms = timer.ElapsedMs();
        }
      }
    }
  }
  stream->SetInputFinished();
  while (stream->DecodeReady()) {
    stream->Decode();
    ++chunks;
  }
  const auto result = stream->GetFinalResult();
  const double elapsed_ms = timer.ElapsedMs();
  const double audio_ms =
      1000.0 * wav_data.samples.size() / wav_data.sample_rate;
  std::cout << "text: " << result.text << "\n";
  std::cout << "sample_rate: " << wav_data.sample_rate << "\n";
  std::cout << "audio_samples: " << wav_data.samples.size() << "\n";
  std::cout << "chunk_ms: " << chunk_ms << "\n";
  std::cout << "chunk_samples: " << chunk_samples << "\n";
  std::cout << "chunks: " << chunks << "\n";
  std::cout << "partial_updates: " << partial_updates << "\n";
  std::cout << "first_partial_latency_ms: " << first_partial_latency_ms << "\n";
  std::cout << "elapsed_ms: " << elapsed_ms << "\n";
  std::cout << "audio_ms: " << audio_ms << "\n";
  std::cout << "rtf: " << (audio_ms > 0.0 ? elapsed_ms / audio_ms : 0.0)
            << "\n";
  return 0;
}
