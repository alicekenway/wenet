#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "asr_sdk/asr_engine.h"
#include "audio/wav_reader.h"
#include "sherpa_onnx_wenet/ctc_onnx_backend_factory.h"
#include "sherpa_onnx_wenet/ctc_greedy_decoder.h"
#include "sherpa_onnx_wenet/streaming_ctc_backend.h"
#include "sherpa_onnx_wenet/token_table.h"
#include "sherpa_onnx_wenet/whisper_feature_extractor.h"
#include "utils/json.h"
#include "utils/timer.h"

namespace {

using asr_sdk::internal::sherpa_onnx_wenet::CtcGreedyDecode;
using asr_sdk::internal::sherpa_onnx_wenet::CreateStreamingCtcBackend;
using asr_sdk::internal::sherpa_onnx_wenet::ParseZipformerFeatureType;
using asr_sdk::internal::sherpa_onnx_wenet::StreamingCtcBackend;
using asr_sdk::internal::sherpa_onnx_wenet::TokenTable;
using asr_sdk::internal::sherpa_onnx_wenet::WhisperFeatureExtractor;
using asr_sdk::internal::sherpa_onnx_wenet::WhisperFeatureOptions;
using asr_sdk::internal::sherpa_onnx_wenet::ZipformerFeatureType;

struct Args {
  std::string model_dir;
  std::string metadata;
  std::string wav_parent;
  std::string output_json;
  std::string wav_key;
  std::string text_key;
  std::string debug_log;
  std::string decode_mode = "lm";
  int limit = 0;
  int num_threads = 1;
  int chunk_ms = 0;
  bool debug = false;
};

struct DecodeOutput {
  std::string hyp;
  std::string debug_json;
  double decode_sec = 0.0;
  std::string error;
};

struct EvalRow {
  int index = 0;
  std::string input_json;
  std::string wav_field;
  std::string wav_path;
  std::string ref;
  std::string hyp;
  std::string debug_json;
  std::string error;
  double audio_sec = 0.0;
  double decode_sec = 0.0;
  double rtf = 0.0;
};

struct GreedyContext {
  std::filesystem::path model_path;
  std::filesystem::path tokens_path;
  std::unique_ptr<TokenTable> tokens;
  std::shared_ptr<StreamingCtcBackend> backend_template;
  ZipformerFeatureType feature_type =
      ZipformerFeatureType::kWhisper;
};

void Usage(const char* argv0) {
  std::cerr
      << "usage: " << argv0 << " --model_dir DIR --metadata metadata.jsonl"
      << " --wav_parent DIR --output_json OUT.jsonl"
      << " [--decode_mode lm|greedy] [--limit N] [--num_threads N]"
      << " [--chunk_ms N] [--wav_key KEY] [--text_key KEY]"
      << " [--debug false] [--debug_log PATH]\n";
}

bool ParseBool(const std::string& value) {
  return value == "1" || value == "true" || value == "yes" || value == "on";
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
    if (key == "--model_dir") {
      if (!NeedValue(argc, argv, &i, &args->model_dir)) return false;
    } else if (key == "--metadata" || key == "--test_set") {
      if (!NeedValue(argc, argv, &i, &args->metadata)) return false;
    } else if (key == "--wav_parent" || key == "--wav_root") {
      if (!NeedValue(argc, argv, &i, &args->wav_parent)) return false;
    } else if (key == "--output_json" || key == "--output") {
      if (!NeedValue(argc, argv, &i, &args->output_json)) return false;
    } else if (key == "--wav_key") {
      if (!NeedValue(argc, argv, &i, &args->wav_key)) return false;
    } else if (key == "--text_key") {
      if (!NeedValue(argc, argv, &i, &args->text_key)) return false;
    } else if (key == "--decode_mode" || key == "--mode") {
      if (!NeedValue(argc, argv, &i, &args->decode_mode)) return false;
    } else if (key == "--limit") {
      if (!NeedValue(argc, argv, &i, &value)) return false;
      args->limit = std::max(0, std::atoi(value.c_str()));
    } else if (key == "--num_threads") {
      if (!NeedValue(argc, argv, &i, &value)) return false;
      args->num_threads = std::max(1, std::atoi(value.c_str()));
	    } else if (key == "--chunk_ms") {
	      if (!NeedValue(argc, argv, &i, &value)) return false;
	      args->chunk_ms = std::max(0, std::atoi(value.c_str()));
    } else if (key == "--debug") {
      if (!NeedValue(argc, argv, &i, &value)) return false;
      args->debug = ParseBool(value);
    } else if (key == "--debug_log") {
      if (!NeedValue(argc, argv, &i, &args->debug_log)) return false;
    } else {
      return false;
    }
  }
  if (args->decode_mode != "lm" && args->decode_mode != "greedy") {
    return false;
  }
  if (!args->debug_log.empty() && args->decode_mode == "lm") {
    args->debug = true;
  }
  return !args->model_dir.empty() && !args->metadata.empty() &&
         !args->wav_parent.empty() && !args->output_json.empty();
}

std::string Trim(std::string value) {
  auto not_space = [](unsigned char c) { return !std::isspace(c); };
  value.erase(value.begin(),
              std::find_if(value.begin(), value.end(), not_space));
  value.erase(std::find_if(value.rbegin(), value.rend(), not_space).base(),
              value.end());
  return value;
}

std::string FirstJsonStringValue(const std::string& json,
                                 const std::vector<std::string>& keys) {
  for (const auto& key : keys) {
    const std::string value =
        asr_sdk::internal::FindJsonStringValue(json, key, "");
    if (!value.empty()) {
      return value;
    }
  }
  return "";
}

std::string ResolveWavPath(const std::string& wav_field,
                           const std::filesystem::path& wav_parent) {
  std::filesystem::path path(wav_field);
  if (path.is_absolute()) {
    return path.string();
  }
  return (wav_parent / path).string();
}

std::filesystem::path ResolvePackagePath(const std::filesystem::path& root,
                                         const std::string& value,
                                         const std::string& fallback) {
  std::filesystem::path path(value.empty() ? fallback : value);
  if (path.is_absolute()) {
    return path;
  }
  return root / path;
}

std::string ReadManifest(const std::filesystem::path& model_dir) {
  const std::filesystem::path manifest = model_dir / "sdk_model.json";
  std::ifstream in(manifest);
  if (!in) {
    return "";
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  return buffer.str();
}

std::string JsonNumber(double value) {
  std::ostringstream out;
  out << std::setprecision(10) << value;
  return out.str();
}

std::string CleanLogField(std::string value) {
  for (char& c : value) {
    if (c == '\n' || c == '\r' || c == '#') {
      c = ' ';
    }
  }
  return value;
}

size_t FindMatching(const std::string& text, size_t begin, char open,
                    char close) {
  bool in_string = false;
  bool escaped = false;
  int depth = 0;
  for (size_t i = begin; i < text.size(); ++i) {
    const char c = text[i];
    if (in_string) {
      if (escaped) {
        escaped = false;
      } else if (c == '\\') {
        escaped = true;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
      continue;
    }
    if (c == open) {
      ++depth;
    } else if (c == close) {
      --depth;
      if (depth == 0) {
        return i;
      }
    }
  }
  return std::string::npos;
}

std::vector<std::string> ExtractFinalNbestObjects(
    const std::string& debug_json) {
  std::vector<std::string> objects;
  const size_t key = debug_json.find("\"final_nbest\"");
  if (key == std::string::npos) {
    return objects;
  }
  const size_t array_begin = debug_json.find('[', key);
  if (array_begin == std::string::npos) {
    return objects;
  }
  const size_t array_end = FindMatching(debug_json, array_begin, '[', ']');
  if (array_end == std::string::npos) {
    return objects;
  }
  size_t pos = array_begin + 1;
  while (pos < array_end) {
    const size_t object_begin = debug_json.find('{', pos);
    if (object_begin == std::string::npos || object_begin >= array_end) {
      break;
    }
    const size_t object_end = FindMatching(debug_json, object_begin, '{', '}');
    if (object_end == std::string::npos || object_end > array_end) {
      break;
    }
    objects.push_back(debug_json.substr(object_begin, object_end - object_begin + 1));
    pos = object_end + 1;
  }
  return objects;
}

std::filesystem::path DefaultDebugLogPath(const std::string& output_json) {
  std::filesystem::path path(output_json);
  path.replace_extension(".debug.txt");
  return path;
}

void WriteDebugLogBlock(std::ostream* out, const EvalRow& row) {
  if (out == nullptr || row.debug_json.empty()) {
    return;
  }
  *out << "ref#" << CleanLogField(row.wav_path) << "#"
       << CleanLogField(row.ref) << "\n";
  const std::vector<std::string> hyps =
      ExtractFinalNbestObjects(row.debug_json);
  if (hyps.empty()) {
    *out << "hyp1#" << CleanLogField(row.hyp) << "##\n\n";
    return;
  }
  for (size_t i = 0; i < hyps.size(); ++i) {
    const std::string text =
        asr_sdk::internal::FindJsonStringValue(hyps[i], "text", "");
    const double am_score =
        asr_sdk::internal::FindJsonDoubleValue(hyps[i], "am_score", 0.0);
    const double lm_score =
        asr_sdk::internal::FindJsonDoubleValue(hyps[i], "lm_score", 0.0);
    *out << "hyp" << (i + 1) << "#" << CleanLogField(text) << "#"
         << JsonNumber(am_score) << "#" << JsonNumber(lm_score) << "\n";
  }
  *out << "\n";
}

std::string AppendJsonFields(const EvalRow& row,
                             const std::string& decode_mode) {
  std::string json = Trim(row.input_json);
  if (json.empty() || json.back() != '}') {
    json = "{}";
  } else {
    json.pop_back();
  }
  if (json.size() > 1) {
    json += ",";
  }
  json += "\"hyp\":\"" + asr_sdk::internal::JsonEscape(row.hyp) + "\"";
  json += ",\"rtf\":" + JsonNumber(row.rtf);
  json += ",\"atf\":" + JsonNumber(row.rtf);
  json += ",\"decode_sec\":" + JsonNumber(row.decode_sec);
  json += ",\"audio_sec\":" + JsonNumber(row.audio_sec);
  json += ",\"decode_mode\":\"" + asr_sdk::internal::JsonEscape(decode_mode) +
          "\"";
  if (!row.error.empty()) {
    json += ",\"error\":\"" + asr_sdk::internal::JsonEscape(row.error) + "\"";
  }
  json += "}";
  return json;
}

void AppendFrame(const std::vector<float>& frame, std::vector<float>* flat) {
  flat->insert(flat->end(), frame.begin(), frame.end());
}

std::vector<std::vector<float>> RunGreedyStreaming(
    StreamingCtcBackend* backend,
    const std::vector<std::vector<float>>& features) {
  const auto& info = backend->Info();
  const int window = info.input_window_frames;
  const int shift = info.input_shift_frames;
  const int dim = info.feature_dim;
  if (window < shift || shift <= 0 || dim <= 0) {
    throw std::runtime_error("invalid CTC chunk geometry");
  }

  std::vector<std::vector<float>> all_log_probs;
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
    for (int frame = shift; frame < window; ++frame) {
      std::vector<float> copy(static_cast<size_t>(dim));
      std::copy(chunk.begin() + static_cast<size_t>(frame * dim),
                chunk.begin() + static_cast<size_t>((frame + 1) * dim),
                copy.begin());
      overlap.push_back(std::move(copy));
    }

    std::vector<std::vector<float>> log_probs;
    backend->Forward(chunk.data(), window, &log_probs);
    all_log_probs.insert(all_log_probs.end(), log_probs.begin(),
                         log_probs.end());
  }
  return all_log_probs;
}

GreedyContext MakeGreedyContext(const std::filesystem::path& model_dir,
                                int num_threads) {
  const std::string manifest_json = ReadManifest(model_dir);
  const std::filesystem::path model_path = ResolvePackagePath(
      model_dir,
      asr_sdk::internal::FindJsonStringValue(manifest_json, "model_path", ""),
      "model.onnx");
  const std::filesystem::path tokens_path = ResolvePackagePath(
      model_dir,
      asr_sdk::internal::FindJsonStringValue(manifest_json, "tokens", ""),
      "tokens.txt");
  const ZipformerFeatureType feature_type = ParseZipformerFeatureType(
      asr_sdk::internal::FindJsonStringValue(manifest_json, "feature_type",
                                             "whisper"));

  GreedyContext context;
  context.model_path = model_path;
  context.tokens_path = tokens_path;
  context.feature_type = feature_type;
  context.tokens = std::make_unique<TokenTable>(tokens_path);
  auto backend_template = CreateStreamingCtcBackend(
      model_path.string(), num_threads, context.tokens->BlankId());
  if (backend_template->Info().vocab_size !=
      context.tokens->ModelVocabSize()) {
    throw std::runtime_error("ONNX vocab size does not match tokens.txt");
  }
  context.backend_template =
      std::shared_ptr<StreamingCtcBackend>(std::move(backend_template));
  return context;
}

DecodeOutput DecodeOne(asr_sdk::AsrEngine* engine,
                       const asr_sdk::internal::WavData& wav,
                       int chunk_ms) {
  DecodeOutput out;
  auto stream_or = engine->CreateStream();
  if (!stream_or.ok()) {
    out.error = stream_or.status().ToString();
    return out;
  }
  auto stream = std::move(stream_or).value();

  asr_sdk::internal::Timer timer;
  asr_sdk::Status status = asr_sdk::Status::Ok();
  if (chunk_ms <= 0) {
    status = stream->AcceptPcm16(wav.samples.data(), wav.samples.size(),
                                 wav.sample_rate);
    if (status.ok()) {
      status = stream->SetInputFinished();
    }
    while (status.ok() && stream->DecodeReady()) {
      status = stream->Decode();
    }
  } else {
    const size_t chunk_samples =
        static_cast<size_t>(wav.sample_rate * chunk_ms / 1000);
    for (size_t offset = 0; status.ok() && offset < wav.samples.size();
         offset += chunk_samples) {
      const size_t n = std::min(chunk_samples, wav.samples.size() - offset);
      status = stream->AcceptPcm16(wav.samples.data() + offset, n,
                                   wav.sample_rate);
      while (status.ok() && stream->DecodeReady()) {
        status = stream->Decode();
      }
    }
    if (status.ok()) {
      status = stream->SetInputFinished();
    }
    while (status.ok() && stream->DecodeReady()) {
      status = stream->Decode();
    }
  }
  out.decode_sec = timer.ElapsedSeconds();
  if (!status.ok()) {
    out.error = status.ToString();
    return out;
  }
  const auto final_result = stream->GetFinalResult();
  out.hyp = final_result.text;
  out.debug_json = final_result.raw_backend_json;
  return out;
}

DecodeOutput DecodeOneGreedy(GreedyContext* context,
                             const asr_sdk::internal::WavData& wav) {
  DecodeOutput out;
  try {
    asr_sdk::internal::Timer timer;
    WhisperFeatureOptions feature_options;
    feature_options.feature_type = context->feature_type;
    WhisperFeatureExtractor extractor(feature_options);
    std::vector<std::vector<float>> features;
    extractor.ExtractPcm16(wav.samples.data(), wav.samples.size(), &features);
    auto backend = context->backend_template->CloneStream();
    const auto log_probs = RunGreedyStreaming(backend.get(), features);
    const auto ids = CtcGreedyDecode(log_probs, context->tokens->BlankId());
    out.hyp = context->tokens->DecodeIds(ids);
    out.decode_sec = timer.ElapsedSeconds();
  } catch (const std::exception& e) {
    out.error = e.what();
  }
  return out;
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!ParseArgs(argc, argv, &args)) {
    Usage(argv[0]);
    return 2;
  }

  std::unique_ptr<asr_sdk::AsrEngine> engine;
  std::unique_ptr<GreedyContext> greedy_context;
  if (args.decode_mode == "lm") {
    asr_sdk::EngineConfig config;
    config.model_dir = args.model_dir;
    config.num_threads = args.num_threads;
    config.debug = args.debug;
    auto engine_or = asr_sdk::AsrEngine::Create(config);
    if (!engine_or.ok()) {
      std::cerr << engine_or.status().ToString() << "\n";
      return 1;
    }
    engine = std::move(engine_or).value();
  } else {
    try {
      greedy_context =
          std::make_unique<GreedyContext>(MakeGreedyContext(args.model_dir,
                                                            args.num_threads));
    } catch (const std::exception& e) {
      std::cerr << "failed to initialize greedy decoder: " << e.what()
                << "\n";
      return 1;
    }
  }

  std::ifstream metadata(args.metadata);
  if (!metadata) {
    std::cerr << "failed to open metadata: " << args.metadata << "\n";
    return 1;
  }
  const auto output_parent = std::filesystem::path(args.output_json).parent_path();
  if (!output_parent.empty()) {
    std::filesystem::create_directories(output_parent);
  }
  std::ofstream output(args.output_json);
  if (!output) {
    std::cerr << "failed to open output_json: " << args.output_json << "\n";
    return 1;
  }

  std::filesystem::path debug_log_path;
  std::ofstream debug_log;
  if (args.debug && args.decode_mode == "lm") {
    debug_log_path = args.debug_log.empty()
                         ? DefaultDebugLogPath(args.output_json)
                         : std::filesystem::path(args.debug_log);
    const auto debug_parent = debug_log_path.parent_path();
    if (!debug_parent.empty()) {
      std::filesystem::create_directories(debug_parent);
    }
    debug_log.open(debug_log_path);
    if (!debug_log) {
      std::cerr << "failed to open debug_log: " << debug_log_path << "\n";
      return 1;
    }
  }

  const std::filesystem::path wav_parent(args.wav_parent);
  std::string line;
  int failed = 0;
  int count = 0;
  double total_audio_sec = 0.0;
  double total_decode_sec = 0.0;

  int index = 0;
  while (std::getline(metadata, line)) {
    if (Trim(line).empty()) {
      continue;
    }
    if (args.limit > 0 && index >= args.limit) {
      break;
    }

    EvalRow row;
    row.index = index;
    row.input_json = line;
    row.ref = args.text_key.empty()
                  ? FirstJsonStringValue(line, {"text", "sentence", "transcript"})
                  : asr_sdk::internal::FindJsonStringValue(line, args.text_key,
                                                           "");
    row.wav_field =
        args.wav_key.empty()
            ? FirstJsonStringValue(
                  line, {"audiofile_path", "audio_filepath", "wav", "path",
                         "file_name"})
            : asr_sdk::internal::FindJsonStringValue(line, args.wav_key, "");
    row.wav_path = ResolveWavPath(row.wav_field, wav_parent);

    asr_sdk::internal::WavData wav;
    asr_sdk::Status wav_status =
        asr_sdk::internal::ReadWavFile(row.wav_path, &wav);
    if (!wav_status.ok()) {
      row.error = wav_status.ToString();
    } else {
      row.audio_sec =
          static_cast<double>(wav.samples.size()) / wav.sample_rate;
      DecodeOutput decode =
          args.decode_mode == "greedy"
              ? DecodeOneGreedy(greedy_context.get(), wav)
              : DecodeOne(engine.get(), wav, args.chunk_ms);
	      row.hyp = std::move(decode.hyp);
      row.debug_json = std::move(decode.debug_json);
	      row.decode_sec = decode.decode_sec;
	      row.error = std::move(decode.error);
      row.rtf = row.audio_sec > 0.0 ? row.decode_sec / row.audio_sec : 0.0;
    }

    total_audio_sec += row.audio_sec;
    total_decode_sec += row.decode_sec;
    failed += !row.error.empty() ? 1 : 0;

    output << AppendJsonFields(row, args.decode_mode) << "\n";
    if (debug_log.is_open()) {
      WriteDebugLogBlock(&debug_log, row);
    }
    ++index;
    ++count;
    if (index % 100 == 0) {
      output.flush();
      std::cerr << "progress_sentences " << index << "\n";
    }
  }

  std::cerr << "decode_mode " << args.decode_mode << "\n";
  std::cerr << "sentence_count " << count << "\n";
  std::cerr << "failed_count " << failed << "\n";
  std::cerr << "rtf "
            << (total_audio_sec > 0.0 ? total_decode_sec / total_audio_sec
                                      : 0.0)
            << "\n";
  std::cerr << "output_json " << args.output_json << "\n";
  if (debug_log.is_open()) {
    debug_log.flush();
    std::cerr << "debug_log " << debug_log_path.string() << "\n";
  }
  return failed == 0 ? 0 : 1;
}
