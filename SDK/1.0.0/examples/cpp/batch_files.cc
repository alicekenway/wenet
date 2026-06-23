#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "io/wav_reader.h"
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

struct WavEntry {
  std::string id;
  std::filesystem::path path;
};

bool ParseWavListLine(const std::string& line, WavEntry* entry) {
  if (entry == nullptr || line.empty() || line[0] == '#') {
    return false;
  }
  std::istringstream in(line);
  std::vector<std::string> fields;
  std::string field;
  while (in >> field) {
    fields.push_back(field);
  }
  if (fields.empty()) {
    return false;
  }
  if (fields.size() == 1) {
    entry->path = fields[0];
    entry->id = entry->path.stem().string();
    return true;
  }
  entry->id = fields[0];
  entry->path = fields[1];
  return true;
}

wenet_sdk::AsrResult DecodeWav(wenet_sdk::Stream* stream,
                               const wenet_sdk::internal::WavData& wav,
                               int chunk_ms) {
  stream->Reset();
  const size_t chunk_samples =
      static_cast<size_t>(std::max(1, wav.sample_rate * chunk_ms / 1000));
  for (size_t offset = 0; offset < wav.samples.size(); offset += chunk_samples) {
    const size_t n = std::min(chunk_samples, wav.samples.size() - offset);
    stream->AcceptWaveform(wav.sample_rate, wav.samples.data() + offset, n);
    while (stream->DecodeReady()) {
      stream->Decode();
    }
  }
  stream->SetInputFinished();
  while (stream->DecodeReady()) {
    stream->Decode();
  }
  return stream->GetFinalResult();
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model_dir = ArgValue(argc, argv, "--model_dir");
  const std::string wav_list = ArgValue(argc, argv, "--wav_list");
  const int chunk_ms = ArgInt(argc, argv, "--chunk_ms", 100);
  if (model_dir.empty() || wav_list.empty()) {
    std::cerr << "usage: batch_files --model_dir MODEL_DIR --wav_list WAV_LIST "
                 "[--chunk_ms 100]\n";
    return 2;
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

  std::ifstream list(wav_list);
  if (!list) {
    std::cerr << "failed to open wav list: " << wav_list << "\n";
    return 1;
  }

  int failures = 0;
  std::string line;
  while (std::getline(list, line)) {
    WavEntry entry;
    if (!ParseWavListLine(line, &entry)) {
      continue;
    }
    wenet_sdk::internal::WavData wav;
    auto status = wenet_sdk::internal::ReadWavFile(entry.path, &wav);
    if (!status.ok()) {
      std::cerr << entry.id << "\tERROR\t" << status.message() << "\n";
      ++failures;
      continue;
    }
    const auto result = DecodeWav(stream.get(), wav, chunk_ms);
    std::cout << entry.id << "\t" << result.text << "\n";
  }
  return failures == 0 ? 0 : 1;
}
