#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>

#include "asr_sdk/asr_engine.h"
#include "audio/wav_reader.h"
#include "utils/timer.h"

namespace {

struct Args {
  std::string model_dir;
  std::string wav;
  int chunk_ms = 100;
  bool print_partial = true;
};

bool ParseBool(const std::string& value) {
  return value == "1" || value == "true" || value == "yes" ||
         value == "on";
}

void Usage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " --model_dir DIR --wav WAV [--chunk_ms 100]"
               " [--print_partial true]\n";
}

bool ParseArgs(int argc, char** argv, Args* args) {
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto need_value = [&](std::string* out) -> bool {
      if (i + 1 >= argc) {
        return false;
      }
      *out = argv[++i];
      return true;
    };
    if (key == "--model_dir") {
      if (!need_value(&args->model_dir)) return false;
    } else if (key == "--wav") {
      if (!need_value(&args->wav)) return false;
    } else if (key == "--chunk_ms") {
      std::string value;
      if (!need_value(&value)) return false;
      args->chunk_ms = std::max(1, std::atoi(value.c_str()));
    } else if (key == "--print_partial") {
      std::string value;
      if (!need_value(&value)) return false;
      args->print_partial = ParseBool(value);
    } else if (key == "--help" || key == "-h") {
      return false;
    } else {
      std::cerr << "unknown argument: " << key << "\n";
      return false;
    }
  }
  return !args->model_dir.empty() && !args->wav.empty();
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!ParseArgs(argc, argv, &args)) {
    Usage(argv[0]);
    return 2;
  }

  asr_sdk::internal::WavData wav;
  asr_sdk::Status status = asr_sdk::internal::ReadWavFile(args.wav, &wav);
  if (!status.ok()) {
    std::cerr << status.ToString() << "\n";
    return 1;
  }

  asr_sdk::EngineConfig config;
  config.model_dir = args.model_dir;
  config.sample_rate = wav.sample_rate;
  auto engine_or = asr_sdk::AsrEngine::Create(config);
  if (!engine_or.ok()) {
    std::cerr << engine_or.status().ToString() << "\n";
    return 1;
  }
  auto stream_or = std::move(engine_or).value()->CreateStream();
  if (!stream_or.ok()) {
    std::cerr << stream_or.status().ToString() << "\n";
    return 1;
  }
  auto stream = std::move(stream_or).value();

  const size_t chunk_samples =
      static_cast<size_t>(wav.sample_rate * args.chunk_ms / 1000);
  asr_sdk::internal::Timer timer;
  std::string last_partial;
  for (size_t offset = 0; offset < wav.samples.size();
       offset += chunk_samples) {
    const size_t n = std::min(chunk_samples, wav.samples.size() - offset);
    status = stream->AcceptPcm16(wav.samples.data() + offset, n,
                                 wav.sample_rate);
    if (!status.ok()) {
      std::cerr << status.ToString() << "\n";
      return 1;
    }
    while (stream->DecodeReady()) {
      status = stream->Decode();
      if (!status.ok()) {
        std::cerr << status.ToString() << "\n";
        return 1;
      }
      const auto partial = stream->GetResult();
      if (args.print_partial && !partial.text.empty() &&
          partial.text != last_partial && !partial.is_final) {
        last_partial = partial.text;
        std::cout << "[partial] " << partial.text << "\n";
      }
    }
  }

  status = stream->SetInputFinished();
  if (!status.ok()) {
    std::cerr << status.ToString() << "\n";
    return 1;
  }
  while (stream->DecodeReady()) {
    status = stream->Decode();
    if (!status.ok()) {
      std::cerr << status.ToString() << "\n";
      return 1;
    }
  }
  const auto final_result = stream->GetFinalResult();
  const double wall_sec = timer.ElapsedSeconds();
  const double audio_sec =
      static_cast<double>(wav.samples.size()) / wav.sample_rate;
  std::cout << "[final] " << final_result.text << "\n";
  std::cout << "audio_sec: " << audio_sec << "\n";
  std::cout << "wall_sec: " << wall_sec << "\n";
  std::cout << "RTF: " << (wall_sec / audio_sec) << "\n";
  std::cout << "wenet_linkage: static\n";
  std::cout << "onnxruntime_linkage: dynamic\n";
  return 0;
}
