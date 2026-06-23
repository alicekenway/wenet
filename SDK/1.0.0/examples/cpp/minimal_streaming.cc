#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "wenet_sdk/asr_engine.h"

int main(int argc, char** argv) {
  const std::string model_dir = argc > 1 ? argv[1] : "model_example";

  wenet_sdk::EngineConfig config;
  config.model_dir = model_dir;
  config.enable_timestamps = true;

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

  const int sample_rate = 16000;
  std::vector<float> pcm(static_cast<size_t>(sample_rate / 2), 0.0f);
  for (size_t i = 0; i < pcm.size(); ++i) {
    pcm[i] = 0.2f * std::sin(2.0 * 3.14159265358979323846 * 440.0 *
                             static_cast<double>(i) / sample_rate);
  }

  const size_t chunk = static_cast<size_t>(sample_rate / 10);
  for (size_t offset = 0; offset < pcm.size(); offset += chunk) {
    const size_t n = std::min(chunk, pcm.size() - offset);
    stream->AcceptWaveform(sample_rate, pcm.data() + offset, n);
    while (stream->DecodeReady()) {
      stream->Decode();
      const auto partial = stream->GetResult();
      if (!partial.text.empty()) {
        std::cout << "partial: " << partial.text << "\n";
      }
    }
  }
  stream->SetInputFinished();
  while (stream->DecodeReady()) {
    stream->Decode();
  }
  std::cout << "final: " << stream->GetFinalResult().text << "\n";
  return 0;
}
