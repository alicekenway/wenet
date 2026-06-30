#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

#include "asr_sdk/asr_engine.h"
#include "audio/wav_reader.h"
#include "utils/timer.h"

namespace {

struct Args {
  std::string model_dir;
  std::string wav_scp;
  std::string result;
};

void Usage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " --model_dir DIR --wav_scp WAV_SCP --result OUT\n";
}

bool ParseArgs(int argc, char** argv, Args* args) {
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (i + 1 >= argc) {
      return false;
    }
    const std::string value = argv[++i];
    if (key == "--model_dir") {
      args->model_dir = value;
    } else if (key == "--wav_scp") {
      args->wav_scp = value;
    } else if (key == "--result") {
      args->result = value;
    } else {
      return false;
    }
  }
  return !args->model_dir.empty() && !args->wav_scp.empty() &&
         !args->result.empty();
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!ParseArgs(argc, argv, &args)) {
    Usage(argv[0]);
    return 2;
  }

  asr_sdk::EngineConfig config;
  config.model_dir = args.model_dir;
  auto engine_or = asr_sdk::AsrEngine::Create(config);
  if (!engine_or.ok()) {
    std::cerr << engine_or.status().ToString() << "\n";
    return 1;
  }
  auto engine = std::move(engine_or).value();

  std::ifstream wav_scp(args.wav_scp);
  if (!wav_scp) {
    std::cerr << "failed to open wav_scp: " << args.wav_scp << "\n";
    return 1;
  }
  std::ofstream result(args.result);
  if (!result) {
    std::cerr << "failed to open result: " << args.result << "\n";
    return 1;
  }

  asr_sdk::internal::Timer timer;
  std::string line;
  int total = 0;
  int failed = 0;
  double audio_sec = 0.0;
  while (std::getline(wav_scp, line)) {
    if (line.empty()) {
      continue;
    }
    std::istringstream parser(line);
    std::string utt;
    std::string wav_path;
    parser >> utt >> wav_path;
    if (utt.empty() || wav_path.empty()) {
      ++failed;
      continue;
    }
    ++total;
    asr_sdk::internal::WavData wav;
    asr_sdk::Status status = asr_sdk::internal::ReadWavFile(wav_path, &wav);
    if (!status.ok()) {
      result << utt << " <ERROR:" << status.message() << ">\n";
      ++failed;
      continue;
    }
    audio_sec += static_cast<double>(wav.samples.size()) / wav.sample_rate;
    auto stream_or = engine->CreateStream();
    if (!stream_or.ok()) {
      result << utt << " <ERROR:" << stream_or.status().message() << ">\n";
      ++failed;
      continue;
    }
    auto stream = std::move(stream_or).value();
    status = stream->AcceptPcm16(wav.samples.data(), wav.samples.size(),
                                 wav.sample_rate);
    if (status.ok()) {
      status = stream->SetInputFinished();
    }
    while (status.ok() && stream->DecodeReady()) {
      status = stream->Decode();
    }
    if (!status.ok()) {
      result << utt << " <ERROR:" << status.message() << ">\n";
      ++failed;
      continue;
    }
    result << utt << " " << stream->GetFinalResult().text << "\n";
    if (total % 100 == 0) {
      result.flush();
      std::cerr << "progress_utterances " << total << "\n";
    }
  }

  const double wall_sec = timer.ElapsedSeconds();
  std::cerr << "decoded_utterances " << total << "\n";
  std::cerr << "failed_utterances " << failed << "\n";
  std::cerr << "audio_sec " << audio_sec << "\n";
  std::cerr << "wall_sec " << wall_sec << "\n";
  std::cerr << "rtf " << (audio_sec > 0.0 ? wall_sec / audio_sec : 0.0)
            << "\n";
  return failed == 0 ? 0 : 1;
}
