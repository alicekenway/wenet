#include <algorithm>
#include <cctype>
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

std::string Lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](char c) {
    return static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  });
  return value;
}

bool ArgBool(int argc, char** argv, const std::string& name, bool fallback) {
  const std::string value = Lower(ArgValue(argc, argv, name));
  if (value.empty()) {
    return fallback;
  }
  if (value == "1" || value == "true" || value == "yes" || value == "on") {
    return true;
  }
  if (value == "0" || value == "false" || value == "no" || value == "off") {
    return false;
  }
  return fallback;
}

bool ParseDecoderType(const std::string& value,
                      wenet_sdk::DecoderType* decoder_type) {
  if (decoder_type == nullptr) {
    return false;
  }
  const std::string decoder = Lower(value);
  if (decoder.empty() || decoder == "auto") {
    *decoder_type = wenet_sdk::DecoderType::kAuto;
    return true;
  }
  if (decoder == "greedy" || decoder == "greedy_ctc") {
    *decoder_type = wenet_sdk::DecoderType::kGreedyCtc;
    return true;
  }
  if (decoder == "prefix" || decoder == "ctc_prefix") {
    *decoder_type = wenet_sdk::DecoderType::kCtcPrefix;
    return true;
  }
  if (decoder == "wfst" || decoder == "ctc_wfst") {
    *decoder_type = wenet_sdk::DecoderType::kCtcWfst;
    return true;
  }
  return false;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model_dir = ArgValue(argc, argv, "--model_dir");
  const std::string wav = ArgValue(argc, argv, "--wav");
  const int chunk_ms = ArgInt(argc, argv, "--chunk_ms", 100);
  const std::string decoder = ArgValue(argc, argv, "--decoder", "auto");
  const bool print_partial = ArgBool(argc, argv, "--print_partial", true);
  if (model_dir.empty() || wav.empty()) {
    std::cerr << "usage: asr_stream_file --model_dir MODEL_DIR --wav WAV "
                 "[--chunk_ms 100] [--decoder auto|greedy|prefix|wfst] "
                 "[--print_partial true]\n";
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
  config.enable_timestamps = true;
  if (!ParseDecoderType(decoder, &config.decoder_type)) {
    std::cerr << "invalid decoder: " << decoder << "\n";
    return 2;
  }
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
  int decode_chunks = 0;
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
      ++decode_chunks;
      const auto partial = stream->GetResult();
      if (!partial.text.empty()) {
        ++partial_updates;
        if (first_partial_latency_ms < 0.0) {
          first_partial_latency_ms = timer.ElapsedMs();
        }
        if (print_partial) {
          std::cout << "[partial] " << partial.text << "\n";
        }
      }
    }
  }
  stream->SetInputFinished();
  while (stream->DecodeReady()) {
    stream->Decode();
    ++decode_chunks;
  }
  const auto final_result = stream->GetFinalResult();
  const double elapsed_ms = timer.ElapsedMs();
  const double audio_ms =
      1000.0 * wav_data.samples.size() / wav_data.sample_rate;
  std::cout << "[final] " << final_result.text << "\n";
  std::cout << "rtf: " << (audio_ms > 0.0 ? elapsed_ms / audio_ms : 0.0)
            << "\n";
  std::cout << "first_partial_latency_ms: " << first_partial_latency_ms
            << "\n";
  std::cout << "partial_updates: " << partial_updates << "\n";
  std::cout << "decode_chunks: " << decode_chunks << "\n";
  std::cout << "audio_ms: " << audio_ms << "\n";
  std::cout << "elapsed_ms: " << elapsed_ms << "\n";
  return 0;
}
