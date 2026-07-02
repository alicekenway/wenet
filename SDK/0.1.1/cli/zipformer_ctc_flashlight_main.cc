#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "audio/wav_reader.h"
#include "flashlight_decoder/decoded_hypothesis.h"
#include "flashlight_decoder/flashlight_ctc_stream_decoder.h"
#include "flashlight_decoder/flashlight_decoder_resource.h"
#include "sherpa_onnx_wenet/ctc_greedy_decoder.h"
#include "sherpa_onnx_wenet/token_table.h"
#include "sherpa_onnx_wenet/whisper_feature_extractor.h"
#include "sherpa_onnx_wenet/zipformer2_ctc_onnx_backend.h"
#include "utils/timer.h"

namespace {

using asr_sdk::internal::flashlight_decoder::DecodedHypothesis;
using asr_sdk::internal::flashlight_decoder::FlashlightCtcStreamDecoder;
using asr_sdk::internal::flashlight_decoder::FlashlightDecoderOptions;
using asr_sdk::internal::flashlight_decoder::FlashlightDecoderResource;
using asr_sdk::internal::flashlight_decoder::JoinWords;
using asr_sdk::internal::sherpa_onnx_wenet::ParseZipformerFeatureType;
using asr_sdk::internal::sherpa_onnx_wenet::CtcGreedyDecode;
using asr_sdk::internal::sherpa_onnx_wenet::TokenTable;
using asr_sdk::internal::sherpa_onnx_wenet::WhisperFeatureExtractor;
using asr_sdk::internal::sherpa_onnx_wenet::WhisperFeatureOptions;
using asr_sdk::internal::sherpa_onnx_wenet::Zipformer2CtcOnnxBackend;
using asr_sdk::internal::sherpa_onnx_wenet::Zipformer2CtcOnnxResource;

struct Args {
  std::string model;
  std::string tokens;
  std::string words;
  std::string lexicon;
  std::string lm;
  std::string mapping;
  std::string final_mapping;
  std::string wav;
  std::string blank_token = "<blk>";
  std::string sil_token = "▁";
  std::string unk_word = "<unk>";
  std::string word_separator;
  std::string feature_type = "whisper";
  int initial_padding_ms = 300;
  int final_padding_ms = 800;
  int num_threads = 1;
  bool print_greedy = true;
  bool print_partial = false;
  FlashlightDecoderOptions decoder;
};

bool ParseBool(const std::string& value) {
  return value == "1" || value == "true" || value == "yes" || value == "on";
}

void Usage(const char* argv0) {
  std::cerr
      << "usage: " << argv0
      << " --model model.onnx --tokens tokens.txt --words words.txt"
	         " --lexicon lexicon.txt --lm lm.bin --wav test.wav"
	         " [--mapping output_mapping.txt]"
         " [--final_mapping final_output_mapping.txt] [--blank_token <blk>]"
         " [--sil_token ▁] [--beam_size 50] [--beam_size_token 20]"
         " [--beam_threshold 25] [--lm_weight 1.5]"
         " [--word_score -0.5] [--unk_score -5] [--sil_score 0]"
         " [--nbest 1] [--num_threads 1] [--feature_type whisper|kaldi]\n";
}

bool NeedValue(int argc, char** argv, int* i, std::string* out) {
  if (*i + 1 >= argc) {
    return false;
  }
  *out = argv[++(*i)];
  return true;
}

bool ParseArgs(int argc, char** argv, Args* args) {
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    std::string value;
    if (key == "--model") {
      if (!NeedValue(argc, argv, &i, &args->model)) return false;
    } else if (key == "--tokens") {
      if (!NeedValue(argc, argv, &i, &args->tokens)) return false;
    } else if (key == "--words") {
      if (!NeedValue(argc, argv, &i, &args->words)) return false;
    } else if (key == "--lexicon") {
      if (!NeedValue(argc, argv, &i, &args->lexicon)) return false;
    } else if (key == "--lm") {
      if (!NeedValue(argc, argv, &i, &args->lm)) return false;
	    } else if (key == "--mapping") {
	      if (!NeedValue(argc, argv, &i, &args->mapping)) return false;
    } else if (key == "--final_mapping") {
      if (!NeedValue(argc, argv, &i, &args->final_mapping)) return false;
	    } else if (key == "--wav") {
      if (!NeedValue(argc, argv, &i, &args->wav)) return false;
    } else if (key == "--blank_token") {
      if (!NeedValue(argc, argv, &i, &args->blank_token)) return false;
    } else if (key == "--sil_token") {
      if (!NeedValue(argc, argv, &i, &args->sil_token)) return false;
    } else if (key == "--unk_word") {
      if (!NeedValue(argc, argv, &i, &args->unk_word)) return false;
    } else if (key == "--word_separator") {
      if (!NeedValue(argc, argv, &i, &args->word_separator)) return false;
    } else if (key == "--feature_type") {
      if (!NeedValue(argc, argv, &i, &args->feature_type)) return false;
    } else if (key == "--initial_padding_ms") {
      if (!NeedValue(argc, argv, &i, &value)) return false;
      args->initial_padding_ms = std::max(0, std::atoi(value.c_str()));
    } else if (key == "--final_padding_ms") {
      if (!NeedValue(argc, argv, &i, &value)) return false;
      args->final_padding_ms = std::max(0, std::atoi(value.c_str()));
    } else if (key == "--num_threads") {
      if (!NeedValue(argc, argv, &i, &value)) return false;
      args->num_threads = std::max(1, std::atoi(value.c_str()));
    } else if (key == "--print_greedy" || key == "--print-greedy") {
      if (!NeedValue(argc, argv, &i, &value)) return false;
      args->print_greedy = ParseBool(value);
    } else if (key == "--print_partial" || key == "--print-partial") {
      if (!NeedValue(argc, argv, &i, &value)) return false;
      args->print_partial = ParseBool(value);
    } else if (key == "--beam_size") {
      if (!NeedValue(argc, argv, &i, &value)) return false;
      args->decoder.beam_size = std::max(1, std::atoi(value.c_str()));
    } else if (key == "--beam_size_token") {
      if (!NeedValue(argc, argv, &i, &value)) return false;
      args->decoder.beam_size_token = std::max(1, std::atoi(value.c_str()));
    } else if (key == "--beam_threshold") {
      if (!NeedValue(argc, argv, &i, &value)) return false;
      args->decoder.beam_threshold = std::atof(value.c_str());
    } else if (key == "--lm_weight") {
      if (!NeedValue(argc, argv, &i, &value)) return false;
      args->decoder.lm_weight = std::atof(value.c_str());
    } else if (key == "--word_score") {
      if (!NeedValue(argc, argv, &i, &value)) return false;
      args->decoder.word_score = std::atof(value.c_str());
    } else if (key == "--unk_score") {
      if (!NeedValue(argc, argv, &i, &value)) return false;
      args->decoder.unk_score = std::atof(value.c_str());
    } else if (key == "--sil_score") {
      if (!NeedValue(argc, argv, &i, &value)) return false;
      args->decoder.sil_score = std::atof(value.c_str());
    } else if (key == "--nbest") {
      if (!NeedValue(argc, argv, &i, &value)) return false;
      args->decoder.nbest = std::max(1, std::atoi(value.c_str()));
    } else if (key == "--log_add" || key == "--log-add") {
      if (!NeedValue(argc, argv, &i, &value)) return false;
      args->decoder.log_add = ParseBool(value);
    } else if (key == "--allow_unk" || key == "--allow-unk") {
      if (!NeedValue(argc, argv, &i, &value)) return false;
      args->decoder.allow_unk = ParseBool(value);
    } else if (key == "--smearing") {
      if (!NeedValue(argc, argv, &i, &args->decoder.smearing)) return false;
    } else if (key == "--help" || key == "-h") {
      return false;
    } else {
      std::cerr << "unknown argument: " << key << "\n";
      return false;
    }
  }

  return !args->model.empty() && !args->tokens.empty() &&
         !args->words.empty() && !args->lexicon.empty() &&
         !args->lm.empty() && !args->wav.empty();
}

void AppendFrame(const std::vector<float>& frame, std::vector<float>* flat) {
  flat->insert(flat->end(), frame.begin(), frame.end());
}

void FlattenFrames(const std::vector<std::vector<float>>& frames,
                   std::vector<float>* flat) {
  flat->clear();
  if (frames.empty()) {
    return;
  }
  flat->reserve(frames.size() * frames[0].size());
  for (const auto& frame : frames) {
    flat->insert(flat->end(), frame.begin(), frame.end());
  }
}

std::vector<std::vector<float>> RunStreaming(
    Zipformer2CtcOnnxBackend* backend,
    FlashlightCtcStreamDecoder* decoder,
    const std::vector<std::vector<float>>& features, int* num_chunks,
    float* output_frame_shift_ms, double* search_sec, bool print_partial,
    const std::string& word_separator) {
  const auto& info = backend->Info();
  const int window = info.input_window_frames;
  const int shift = info.input_shift_frames;
  const int dim = info.feature_dim;
  if (window <= shift || shift <= 0 || dim <= 0) {
    throw std::runtime_error("invalid Zipformer chunk geometry");
  }

  std::vector<std::vector<float>> all_log_probs;
  std::vector<std::vector<float>> overlap;
  size_t pos = 0;
  bool first = true;
  *num_chunks = 0;
  *output_frame_shift_ms = 0.0f;
  *search_sec = 0.0;

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
    for (int frame = shift; frame < window; ++frame) {
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
    std::vector<float> flat;
    FlattenFrames(log_probs, &flat);
    asr_sdk::internal::Timer search_timer;
    asr_sdk::Status status = decoder->DecodeChunk(
        flat.data(), static_cast<int>(log_probs.size()), info.vocab_size);
    *search_sec += search_timer.ElapsedSeconds();
    if (!status.ok()) {
      throw std::runtime_error(status.ToString());
    }
    if (print_partial) {
      auto partial = decoder->PartialResult();
      if (partial.ok()) {
        std::cout << "partial mapped words: "
                  << JoinWords(partial.value().mapped_words, " ") << "\n";
        std::cout << "partial mapped text: "
                  << JoinWords(partial.value().mapped_words, word_separator)
                  << "\n";
      }
    }
    all_log_probs.insert(all_log_probs.end(), log_probs.begin(),
                         log_probs.end());
    ++(*num_chunks);
  }
  return all_log_probs;
}

void PrintHypothesis(const DecodedHypothesis& hyp, int index,
                     const std::string& word_separator) {
  std::cout << "hyp " << index << " raw words: "
            << JoinWords(hyp.raw_words, " ") << "\n";
  std::cout << "hyp " << index << " mapped words: "
            << JoinWords(hyp.mapped_words, " ") << "\n";
  std::cout << "hyp " << index << " mapped text: "
            << JoinWords(hyp.mapped_words, word_separator) << "\n";
  std::cout << "hyp " << index << " total score: " << hyp.total_score << "\n";
  std::cout << "hyp " << index << " AM score: " << hyp.am_score << "\n";
  std::cout << "hyp " << index << " LM score: " << hyp.lm_score << "\n";
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
      std::cerr << "Zipformer CTC Flashlight tool expects 16 kHz mono PCM16 WAV\n";
      return 1;
    }

    WhisperFeatureOptions feature_options;
    feature_options.feature_type = ParseZipformerFeatureType(args.feature_type);
    feature_options.initial_padding_ms = args.initial_padding_ms;
    feature_options.final_padding_ms = args.final_padding_ms;
    WhisperFeatureExtractor extractor(feature_options);

    asr_sdk::internal::Timer total_timer;
    asr_sdk::internal::Timer feature_timer;
    std::vector<std::vector<float>> features;
    extractor.ExtractPcm16(wav.samples.data(), wav.samples.size(), &features);
    const double feature_sec = feature_timer.ElapsedSeconds();

    auto acoustic_resource = std::make_shared<Zipformer2CtcOnnxResource>(
        args.model, args.num_threads, tokens.BlankId());
    if (acoustic_resource->Info().vocab_size != tokens.ModelVocabSize()) {
      throw std::runtime_error("ONNX vocab size does not match tokens.txt");
    }
    Zipformer2CtcOnnxBackend backend(acoustic_resource);

	    auto decoder_resource = std::make_shared<FlashlightDecoderResource>(
	        args.tokens, args.words, args.lexicon, args.lm, args.mapping,
	        args.final_mapping, args.decoder, args.blank_token, args.sil_token,
	        args.unk_word);
    FlashlightCtcStreamDecoder decoder(decoder_resource);

    int num_chunks = 0;
    float output_frame_shift_ms = 0.0f;
    double search_sec = 0.0;
    asr_sdk::internal::Timer forward_timer;
    std::vector<std::vector<float>> log_probs = RunStreaming(
        &backend, &decoder, features, &num_chunks, &output_frame_shift_ms,
        &search_sec, args.print_partial, args.word_separator);
    const double forward_plus_search_sec = forward_timer.ElapsedSeconds();
    const double forward_sec = std::max(0.0, forward_plus_search_sec - search_sec);

    std::vector<int> greedy_ids;
    std::string greedy_text;
    if (args.print_greedy) {
      greedy_ids = CtcGreedyDecode(log_probs, tokens.BlankId());
      greedy_text = tokens.DecodeIds(greedy_ids);
    }

    asr_sdk::internal::Timer finalize_timer;
    auto final = decoder.Finalize();
    search_sec += finalize_timer.ElapsedSeconds();
    if (!final.ok()) {
      throw std::runtime_error(final.status().ToString());
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
    const auto& hyps = final.value();
    for (int i = 0; i < static_cast<int>(hyps.size()); ++i) {
      PrintHypothesis(hyps[static_cast<size_t>(i)], i, args.word_separator);
    }
    std::cout << "feature frames: " << features.size() << "\n";
    std::cout << "number of CTC frames: " << log_probs.size() << "\n";
    std::cout << "chunks: " << num_chunks << "\n";
    std::cout << "output_frame_shift_ms: " << output_frame_shift_ms << "\n";
    std::cout << "lexicon entries: "
              << decoder_resource->LexiconEntryCount() << "\n";
    std::cout << "am mapping rules: "
              << decoder_resource->AmMapper().RuleCount() << "\n";
    std::cout << "final mapping rules: "
              << decoder_resource->FinalMapper().RuleCount() << "\n";
    std::cout << "feature RTF: " << feature_sec / audio_sec << "\n";
    std::cout << "forward RTF: " << forward_sec / audio_sec << "\n";
    std::cout << "search RTF: " << search_sec / audio_sec << "\n";
    std::cout << "total RTF: " << total_sec / audio_sec << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
