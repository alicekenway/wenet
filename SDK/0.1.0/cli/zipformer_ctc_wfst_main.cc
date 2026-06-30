#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "audio/wav_reader.h"
#include "decoder/ctc_wfst_beam_search.h"
#include "fst/fstlib.h"
#include "sherpa_onnx_wenet/ctc_greedy_decoder.h"
#include "sherpa_onnx_wenet/token_table.h"
#include "sherpa_onnx_wenet/whisper_feature_extractor.h"
#include "sherpa_onnx_wenet/zipformer2_ctc_onnx_backend.h"
#include "utils/timer.h"

namespace {

using asr_sdk::internal::sherpa_onnx_wenet::CtcGreedyDecode;
using asr_sdk::internal::sherpa_onnx_wenet::TokenTable;
using asr_sdk::internal::sherpa_onnx_wenet::WhisperFeatureExtractor;
using asr_sdk::internal::sherpa_onnx_wenet::WhisperFeatureOptions;
using asr_sdk::internal::sherpa_onnx_wenet::Zipformer2CtcOnnxBackend;
using asr_sdk::internal::sherpa_onnx_wenet::Zipformer2CtcOnnxResource;

struct Args {
  std::string model;
  std::string tokens;
  std::string fst;
  std::string words;
  std::string wav;
  int initial_padding_ms = 300;
  int final_padding_ms = 800;
  int num_threads = 1;
  bool print_greedy = true;
  bool print_wfst = false;
};

class WordSymbolTable {
 public:
  explicit WordSymbolTable(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
      throw std::runtime_error("failed to open words.txt: " + path);
    }
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty()) continue;
      const size_t last_space = line.find_last_of(" \t");
      if (last_space == std::string::npos) continue;
      const int id = std::stoi(line.substr(last_space + 1));
      std::string symbol = line.substr(0, last_space);
      while (!symbol.empty() &&
             (symbol.back() == ' ' || symbol.back() == '\t')) {
        symbol.pop_back();
      }
      if (id >= static_cast<int>(id_to_symbol_.size())) {
        id_to_symbol_.resize(static_cast<size_t>(id + 1));
      }
      id_to_symbol_[static_cast<size_t>(id)] = symbol;
    }
  }

  std::string DecodeIds(const std::vector<int>& ids) const {
    std::string text;
    for (int id : ids) {
      if (id <= 0 || id >= static_cast<int>(id_to_symbol_.size())) {
        continue;
      }
      const std::string& symbol = id_to_symbol_[static_cast<size_t>(id)];
      if (symbol.empty() || symbol[0] == '<' || symbol[0] == '#') {
        continue;
      }
      text += symbol;
    }
    return text;
  }

 private:
  std::vector<std::string> id_to_symbol_;
};

bool ParseBool(const std::string& value) {
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

void Usage(const char* argv0) {
  std::cerr
      << "usage: " << argv0
      << " --model model.onnx --tokens tokens.txt --wav test.wav"
         " [--fst TLG.fst --words words.txt --print-wfst true]"
         " [--initial_padding_ms 300] [--final_padding_ms 800]"
         " [--num_threads 1]\n";
}

bool ParseArgs(int argc, char** argv, Args* args) {
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto need_value = [&](std::string* out) -> bool {
      if (i + 1 >= argc) return false;
      *out = argv[++i];
      return true;
    };
    if (key == "--model") {
      if (!need_value(&args->model)) return false;
    } else if (key == "--tokens") {
      if (!need_value(&args->tokens)) return false;
    } else if (key == "--fst") {
      if (!need_value(&args->fst)) return false;
    } else if (key == "--words") {
      if (!need_value(&args->words)) return false;
    } else if (key == "--wav") {
      if (!need_value(&args->wav)) return false;
    } else if (key == "--initial_padding_ms") {
      std::string value;
      if (!need_value(&value)) return false;
      args->initial_padding_ms = std::max(0, std::atoi(value.c_str()));
    } else if (key == "--final_padding_ms") {
      std::string value;
      if (!need_value(&value)) return false;
      args->final_padding_ms = std::max(0, std::atoi(value.c_str()));
    } else if (key == "--num_threads") {
      std::string value;
      if (!need_value(&value)) return false;
      args->num_threads = std::max(1, std::atoi(value.c_str()));
    } else if (key == "--print-greedy" || key == "--print_greedy") {
      std::string value;
      if (!need_value(&value)) return false;
      args->print_greedy = ParseBool(value);
    } else if (key == "--print-wfst" || key == "--print_wfst") {
      std::string value;
      if (!need_value(&value)) return false;
      args->print_wfst = ParseBool(value);
    } else if (key == "--help" || key == "-h") {
      return false;
    } else {
      std::cerr << "unknown argument: " << key << "\n";
      return false;
    }
  }
  if (args->print_wfst && args->fst.empty()) {
    std::cerr << "--print-wfst requires --fst\n";
    return false;
  }
  return !args->model.empty() && !args->tokens.empty() && !args->wav.empty();
}

void AppendFrame(const std::vector<float>& frame, std::vector<float>* flat) {
  flat->insert(flat->end(), frame.begin(), frame.end());
}

void RunStreaming(
    Zipformer2CtcOnnxBackend* backend,
    const std::vector<std::vector<float>>& features,
    std::vector<std::vector<float>>* all_log_probs, int* num_chunks,
    float* output_frame_shift_ms) {
  const auto& info = backend->Info();
  const int window = info.input_window_frames;
  const int shift = info.input_shift_frames;
  const int dim = info.feature_dim;
  if (window <= shift || shift <= 0 || dim <= 0) {
    throw std::runtime_error("invalid Zipformer chunk geometry");
  }
  all_log_probs->clear();
  *num_chunks = 0;
  *output_frame_shift_ms = 0.0f;
  if (features.empty()) {
    return;
  }

  std::vector<std::vector<float>> overlap;
  size_t pos = 0;
  bool first = true;
  while (pos < features.size() || first) {
    std::vector<float> chunk;
    chunk.reserve(static_cast<size_t>(window * dim));
    if (first) {
      const size_t take =
          std::min(features.size(), static_cast<size_t>(window));
      for (size_t i = 0; i < take; ++i) {
        AppendFrame(features[i], &chunk);
      }
      pos = take;
      first = false;
    } else {
      for (const auto& frame : overlap) {
        AppendFrame(frame, &chunk);
      }
      const size_t take =
          std::min(features.size() - pos, static_cast<size_t>(shift));
      for (size_t i = 0; i < take; ++i) {
        AppendFrame(features[pos + i], &chunk);
      }
      pos += take;
    }

    const int frames_in_chunk = static_cast<int>(chunk.size() / dim);
    if (frames_in_chunk < window) {
      chunk.insert(chunk.end(),
                   static_cast<size_t>((window - frames_in_chunk) * dim),
                   0.0f);
    }

    overlap.clear();
    overlap.reserve(static_cast<size_t>(window - shift));
    const int overlap_begin = shift;
    for (int frame = overlap_begin; frame < window; ++frame) {
      std::vector<float> copy(static_cast<size_t>(dim));
      std::copy(chunk.begin() + static_cast<size_t>(frame * dim),
                chunk.begin() + static_cast<size_t>((frame + 1) * dim),
                copy.begin());
      overlap.push_back(std::move(copy));
    }

    std::vector<std::vector<float>> log_probs;
    backend->Forward(chunk.data(), window, &log_probs);
    if (*num_chunks == 0 && !log_probs.empty()) {
      *output_frame_shift_ms =
          static_cast<float>(shift * 10.0 / log_probs.size());
    }
    all_log_probs->insert(all_log_probs->end(), log_probs.begin(),
                          log_probs.end());
    ++(*num_chunks);
  }
}

std::vector<int> RunWfst(const std::string& fst_path,
                         const std::vector<std::vector<float>>& log_probs,
                         int blank_id) {
  std::unique_ptr<fst::VectorFst<fst::StdArc>> fst(
      fst::VectorFst<fst::StdArc>::Read(fst_path));
  if (!fst) {
    throw std::runtime_error("failed to read FST: " + fst_path);
  }

  wenet::CtcWfstBeamSearchOptions opts;
  opts.max_active = 7000;
  opts.min_active = 200;
  opts.beam = 16.0;
  opts.lattice_beam = 10.0;
  opts.acoustic_scale = 1.0;
  opts.blank = blank_id;
  opts.blank_skip_thresh = 1.0;
  opts.blank_scale = 1.0;
  opts.nbest = 1;

  wenet::CtcWfstBeamSearch searcher(*fst, opts, nullptr);
  searcher.Search(log_probs);
  searcher.FinalizeSearch();
  if (searcher.Outputs().empty()) {
    return {};
  }
  return searcher.Outputs().front();
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!ParseArgs(argc, argv, &args)) {
    Usage(argv[0]);
    return 2;
  }

  try {
    TokenTable tokens(args.tokens);
    asr_sdk::internal::WavData wav;
    asr_sdk::Status status = asr_sdk::internal::ReadWavFile(args.wav, &wav);
    if (!status.ok()) {
      std::cerr << status.ToString() << "\n";
      return 1;
    }
    if (wav.sample_rate != 16000 || wav.num_channels != 1) {
      std::cerr << "Zipformer CTC smoke tool expects 16 kHz mono PCM16 WAV\n";
      return 1;
    }

    WhisperFeatureOptions feature_options;
    feature_options.initial_padding_ms = args.initial_padding_ms;
    feature_options.final_padding_ms = args.final_padding_ms;
    WhisperFeatureExtractor extractor(feature_options);

    asr_sdk::internal::Timer total_timer;
    asr_sdk::internal::Timer feature_timer;
    std::vector<std::vector<float>> features;
    extractor.ExtractPcm16(wav.samples.data(), wav.samples.size(), &features);
    const double feature_sec = feature_timer.ElapsedSeconds();

    auto resource = std::make_shared<Zipformer2CtcOnnxResource>(
        args.model, args.num_threads, tokens.BlankId());
    if (resource->Info().vocab_size != tokens.ModelVocabSize()) {
      throw std::runtime_error("ONNX vocab size does not match tokens.txt");
    }
    Zipformer2CtcOnnxBackend backend(resource);

    std::vector<std::vector<float>> log_probs;
    int num_chunks = 0;
    float output_frame_shift_ms = 0.0f;
    asr_sdk::internal::Timer forward_timer;
    RunStreaming(&backend, features, &log_probs, &num_chunks,
                 &output_frame_shift_ms);
    const double forward_sec = forward_timer.ElapsedSeconds();

    std::vector<int> greedy_ids;
    std::string greedy_text;
    if (args.print_greedy) {
      greedy_ids = CtcGreedyDecode(log_probs, tokens.BlankId());
      greedy_text = tokens.DecodeIds(greedy_ids);
    }

    double search_sec = 0.0;
    std::string wfst_text;
    if (args.print_wfst) {
      asr_sdk::internal::Timer search_timer;
      const std::vector<int> wfst_ids =
          RunWfst(args.fst, log_probs, tokens.BlankId());
      search_sec = search_timer.ElapsedSeconds();
      if (!args.words.empty()) {
        WordSymbolTable words(args.words);
        wfst_text = words.DecodeIds(wfst_ids);
      } else {
        wfst_text = tokens.DecodeIds(wfst_ids);
      }
    }

    const double total_sec = total_timer.ElapsedSeconds();
    const double audio_sec =
        static_cast<double>(wav.samples.size()) / wav.sample_rate;
    if (args.print_greedy) {
      std::cout << "greedy text: " << greedy_text << "\n";
      std::cout << "greedy token ids:";
      for (int id : greedy_ids) std::cout << " " << id;
      std::cout << "\n";
    }
    if (args.print_wfst) {
      std::cout << "WFST text: " << wfst_text << "\n";
    }
    std::cout << "feature frames: " << features.size() << "\n";
    std::cout << "number of CTC frames: " << log_probs.size() << "\n";
    std::cout << "chunks: " << num_chunks << "\n";
    std::cout << "output_frame_shift_ms: " << output_frame_shift_ms << "\n";
    std::cout << "feature RTF: " << feature_sec / audio_sec << "\n";
    std::cout << "forward RTF: " << forward_sec / audio_sec << "\n";
    if (args.print_wfst) {
      std::cout << "search RTF: " << search_sec / audio_sec << "\n";
    }
    std::cout << "total RTF: " << total_sec / audio_sec << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
