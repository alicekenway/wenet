#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "wenet_sdk/asr_engine.h"

namespace {

constexpr int kSampleRate = 16000;
constexpr double kPi = 3.14159265358979323846;

class AudioQueue {
 public:
  void Push(const int16_t* samples, size_t n) {
    if (samples == nullptr || n == 0) {
      return;
    }
    std::lock_guard<std::mutex> lock(mu_);
    chunks_.emplace_back(samples, samples + n);
  }

  bool Pop(std::vector<int16_t>* chunk) {
    if (chunk == nullptr) {
      return false;
    }
    std::lock_guard<std::mutex> lock(mu_);
    if (chunks_.empty()) {
      return false;
    }
    *chunk = std::move(chunks_.front());
    chunks_.pop_front();
    return true;
  }

 private:
  std::mutex mu_;
  std::deque<std::vector<int16_t>> chunks_;
};

void AudioCallback(AudioQueue* queue, const int16_t* samples, size_t n) {
  queue->Push(samples, n);
}

std::vector<int16_t> GeneratePcm16(size_t samples, double hz) {
  std::vector<int16_t> pcm(samples);
  for (size_t i = 0; i < samples; ++i) {
    const double t = static_cast<double>(i) / kSampleRate;
    pcm[i] = static_cast<int16_t>(12000.0 * std::sin(2.0 * kPi * hz * t));
  }
  return pcm;
}

void DecodeQueuedAudio(AudioQueue* queue, wenet_sdk::Stream* stream) {
  std::vector<int16_t> pcm16;
  std::vector<float> pcm_float;
  while (queue->Pop(&pcm16)) {
    pcm_float.resize(pcm16.size());
    for (size_t i = 0; i < pcm16.size(); ++i) {
      pcm_float[i] = static_cast<float>(pcm16[i]) / 32768.0f;
    }
    stream->AcceptWaveform(kSampleRate, pcm_float.data(), pcm_float.size());
    while (stream->DecodeReady()) {
      stream->Decode();
      const auto partial = stream->GetResult();
      if (!partial.text.empty()) {
        std::cout << "partial: " << partial.text << "\n";
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model_dir = argc > 1 ? argv[1] : "model_example";

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

  AudioQueue queue;
  const size_t callback_samples = static_cast<size_t>(kSampleRate / 50);
  const auto pcm = GeneratePcm16(static_cast<size_t>(kSampleRate / 2), 440.0);
  for (size_t offset = 0; offset < pcm.size(); offset += callback_samples) {
    const size_t n = std::min(callback_samples, pcm.size() - offset);
    AudioCallback(&queue, pcm.data() + offset, n);
    DecodeQueuedAudio(&queue, stream.get());
  }

  DecodeQueuedAudio(&queue, stream.get());
  stream->SetInputFinished();
  while (stream->DecodeReady()) {
    stream->Decode();
  }
  std::cout << "final: " << stream->GetFinalResult().text << "\n";
  return 0;
}
