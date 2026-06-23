#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "wenet_sdk/asr_engine.h"

#ifdef WENETSDK_ENABLE_PORTAUDIO
#include "portaudio.h"  // NOLINT
#endif

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

void PrintUsage() {
  std::cerr << "usage: asr_stream_mic --model_dir MODEL_DIR [--seconds 10] "
               "[--sample_rate 16000] [--chunk_ms 100]\n";
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model_dir = ArgValue(argc, argv, "--model_dir");
  const int seconds = ArgInt(argc, argv, "--seconds", 10);
  const int sample_rate = ArgInt(argc, argv, "--sample_rate", 16000);
  const int chunk_ms = ArgInt(argc, argv, "--chunk_ms", 100);
  if (model_dir.empty()) {
    PrintUsage();
    return 2;
  }

#ifndef WENETSDK_ENABLE_PORTAUDIO
  (void)seconds;
  (void)sample_rate;
  (void)chunk_ms;
  std::cerr << "asr_stream_mic was built without PortAudio. Reconfigure with "
               "-DWENETSDK_ENABLE_PORTAUDIO=ON after installing PortAudio.\n";
  return 2;
#else
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

  PaError err = Pa_Initialize();
  if (err != paNoError) {
    std::cerr << "PortAudio init failed: " << Pa_GetErrorText(err) << "\n";
    return 1;
  }

  const unsigned long frames_per_buffer =
      static_cast<unsigned long>(std::max(1, sample_rate * chunk_ms / 1000));
  PaStream* pa_stream = nullptr;
  err = Pa_OpenDefaultStream(&pa_stream, 1, 0, paFloat32, sample_rate,
                             frames_per_buffer, nullptr, nullptr);
  if (err != paNoError) {
    std::cerr << "failed to open microphone: " << Pa_GetErrorText(err)
              << "\n";
    Pa_Terminate();
    return 1;
  }
  err = Pa_StartStream(pa_stream);
  if (err != paNoError) {
    std::cerr << "failed to start microphone: " << Pa_GetErrorText(err)
              << "\n";
    Pa_CloseStream(pa_stream);
    Pa_Terminate();
    return 1;
  }

  std::vector<float> buffer(static_cast<size_t>(frames_per_buffer), 0.0f);
  const int iterations =
      std::max(1, seconds * 1000 / std::max(1, chunk_ms));
  for (int i = 0; i < iterations; ++i) {
    err = Pa_ReadStream(pa_stream, buffer.data(), frames_per_buffer);
    if (err != paNoError) {
      std::cerr << "microphone read failed: " << Pa_GetErrorText(err) << "\n";
      break;
    }
    stream->AcceptWaveform(sample_rate, buffer.data(), buffer.size());
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

  Pa_StopStream(pa_stream);
  Pa_CloseStream(pa_stream);
  Pa_Terminate();
  return 0;
#endif
}
