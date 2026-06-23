#include "model/model_metadata.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

#include "decoder/fst_loader.h"
#include "decoder/symbol_table.h"
#include "frontend/cmvn.h"
#include "utils/checksum.h"

namespace wenet_sdk::internal {
namespace {

Status ReadTextFile(const std::filesystem::path& path, std::string* out) {
  std::ifstream in(path);
  if (!in) {
    return Status::NotFound("failed to open file: " + path.string());
  }
  std::ostringstream ss;
  ss << in.rdbuf();
  *out = ss.str();
  return Status::OK();
}

std::string Lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

std::string ExtractJsonString(const std::string& json, const std::string& key,
                              const std::string& fallback) {
  const std::string needle = "\"" + key + "\"";
  const auto key_pos = json.find(needle);
  if (key_pos == std::string::npos) {
    return fallback;
  }
  const auto colon = json.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return fallback;
  }
  const auto quote = json.find('"', colon + 1);
  if (quote == std::string::npos) {
    return fallback;
  }
  const auto end = json.find('"', quote + 1);
  if (end == std::string::npos) {
    return fallback;
  }
  return json.substr(quote + 1, end - quote - 1);
}

int ExtractJsonInt(const std::string& json, const std::string& key,
                   int fallback) {
  const std::string needle = "\"" + key + "\"";
  const auto key_pos = json.find(needle);
  if (key_pos == std::string::npos) {
    return fallback;
  }
  const auto colon = json.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return fallback;
  }
  const auto begin = json.find_first_of("-0123456789", colon + 1);
  if (begin == std::string::npos) {
    return fallback;
  }
  size_t end = begin;
  while (end < json.size() &&
         (json[end] == '-' || std::isdigit(static_cast<unsigned char>(json[end])))) {
    ++end;
  }
  int parsed = fallback;
  return ParseInteger(json.substr(begin, end - begin), &parsed) ? parsed
                                                               : fallback;
}

float ExtractJsonFloat(const std::string& json, const std::string& key,
                       float fallback) {
  const std::string needle = "\"" + key + "\"";
  const auto key_pos = json.find(needle);
  if (key_pos == std::string::npos) {
    return fallback;
  }
  const auto colon = json.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return fallback;
  }
  const auto begin = json.find_first_of("-0123456789.", colon + 1);
  if (begin == std::string::npos) {
    return fallback;
  }
  size_t end = begin;
  while (end < json.size() &&
         (json[end] == '-' || json[end] == '+' || json[end] == '.' ||
          json[end] == 'e' || json[end] == 'E' ||
          std::isdigit(static_cast<unsigned char>(json[end])))) {
    ++end;
  }
  try {
    return std::stof(json.substr(begin, end - begin));
  } catch (...) {
    return fallback;
  }
}

std::string StripYamlValue(std::string value) {
  value = Trim(std::move(value));
  const auto comment = value.find('#');
  if (comment != std::string::npos) {
    value = Trim(value.substr(0, comment));
  }
  if (value.size() >= 2 &&
      ((value.front() == '"' && value.back() == '"') ||
       (value.front() == '\'' && value.back() == '\''))) {
    value = value.substr(1, value.size() - 2);
  }
  return value;
}

void ParseYamlConfig(const std::string& yaml, ModelMetadata* metadata) {
  std::istringstream iss(yaml);
  std::string line;
  std::string section;
  while (std::getline(iss, line)) {
    const auto comment = line.find('#');
    if (comment != std::string::npos) {
      line = line.substr(0, comment);
    }
    if (Trim(line).empty()) {
      continue;
    }
    const bool is_section = !std::isspace(static_cast<unsigned char>(line[0]));
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    if (is_section) {
      section = Trim(line.substr(0, colon));
      continue;
    }
    const std::string key = Trim(line.substr(0, colon));
    const std::string value = StripYamlValue(line.substr(colon + 1));
    int int_value = 0;
    if (section == "streaming") {
      if (key == "chunk_size" && ParseInteger(value, &int_value)) {
        metadata->streaming.chunk_size = int_value;
      } else if (key == "num_left_chunks" && ParseInteger(value, &int_value)) {
        metadata->streaming.num_left_chunks = int_value;
      } else if (key == "endpoint_silence_ms" &&
                 ParseInteger(value, &int_value)) {
        metadata->streaming.endpoint_silence_ms = int_value;
      } else if (key == "max_segment_ms" && ParseInteger(value, &int_value)) {
        metadata->streaming.max_segment_ms = int_value;
      }
    } else if (section == "runtime") {
      if (key == "intra_op_num_threads" && ParseInteger(value, &int_value)) {
        metadata->runtime.intra_op_num_threads = int_value;
      } else if (key == "inter_op_num_threads" &&
                 ParseInteger(value, &int_value)) {
        metadata->runtime.inter_op_num_threads = int_value;
      } else if (key == "enable_profiling") {
        metadata->runtime.enable_profiling = Lower(value) == "true";
      }
    } else if (section == "postprocess") {
      if (key == "lowercase") {
        metadata->postprocess.lowercase = Lower(value) == "true";
      } else if (key == "remove_bpe_marker") {
        metadata->postprocess.remove_bpe_marker = Lower(value) == "true";
      } else if (key == "language_type") {
        metadata->postprocess.language_type = value;
      }
    }
  }
}

Status ValidateRequiredFile(const std::filesystem::path& path,
                            const std::string& label) {
  if (!std::filesystem::exists(path)) {
    return Status::NotFound(label + " is missing: " + path.string());
  }
  return Status::OK();
}

}  // namespace

DecoderType ParseDecoderType(const std::string& value) {
  const std::string type = Lower(value);
  if (type == "auto") {
    return DecoderType::kAuto;
  }
  if (type == "ctc_wfst" || type == "wfst") {
    return DecoderType::kCtcWfst;
  }
  if (type == "ctc_prefix" || type == "prefix") {
    return DecoderType::kCtcPrefix;
  }
  return DecoderType::kGreedyCtc;
}

std::string DecoderTypeName(DecoderType type) {
  switch (type) {
    case DecoderType::kAuto:
      return "auto";
    case DecoderType::kGreedyCtc:
      return "greedy_ctc";
    case DecoderType::kCtcPrefix:
      return "ctc_prefix";
    case DecoderType::kCtcWfst:
      return "ctc_wfst";
  }
  return "greedy_ctc";
}

Status LoadModelMetadata(const std::filesystem::path& model_dir,
                         ModelMetadata* metadata) {
  if (metadata == nullptr) {
    return Status::InvalidArgument("metadata output is null");
  }
  if (!std::filesystem::exists(model_dir)) {
    return Status::NotFound("model directory does not exist: " +
                            model_dir.string());
  }
  if (!std::filesystem::is_directory(model_dir)) {
    return Status::InvalidArgument("model_dir is not a directory: " +
                                   model_dir.string());
  }

  *metadata = ModelMetadata{};
  metadata->model_dir = model_dir;

  std::string manifest;
  auto status = ReadTextFile(model_dir / "manifest.json", &manifest);
  if (!status.ok()) {
    return status;
  }

  metadata->sdk_model_version =
      ExtractJsonInt(manifest, "sdk_model_version", metadata->sdk_model_version);
  metadata->model_type =
      ExtractJsonString(manifest, "model_type", metadata->model_type);
  metadata->sample_rate =
      ExtractJsonInt(manifest, "sample_rate", metadata->sample_rate);
  metadata->feature_dim =
      ExtractJsonInt(manifest, "feature_dim", metadata->feature_dim);
  metadata->frame_length_ms =
      ExtractJsonInt(manifest, "frame_length_ms", metadata->frame_length_ms);
  metadata->frame_shift_ms =
      ExtractJsonInt(manifest, "frame_shift_ms", metadata->frame_shift_ms);
  metadata->subsampling_rate =
      ExtractJsonInt(manifest, "subsampling_rate", metadata->subsampling_rate);
  metadata->waveform_scale =
      ExtractJsonFloat(manifest, "waveform_scale", metadata->waveform_scale);

  metadata->onnx.encoder =
      ExtractJsonString(manifest, "encoder", metadata->onnx.encoder);
  metadata->onnx.ctc = ExtractJsonString(manifest, "ctc", metadata->onnx.ctc);
  metadata->onnx.output_type =
      ExtractJsonString(manifest, "output_type", metadata->onnx.output_type);
  metadata->onnx.chunk_input_name =
      ExtractJsonString(manifest, "chunk", metadata->onnx.chunk_input_name);
  metadata->onnx.offset_input_name =
      ExtractJsonString(manifest, "offset", metadata->onnx.offset_input_name);
  metadata->onnx.att_cache_input_name = ExtractJsonString(
      manifest, "att_cache", metadata->onnx.att_cache_input_name);
  metadata->onnx.cnn_cache_input_name = ExtractJsonString(
      manifest, "cnn_cache", metadata->onnx.cnn_cache_input_name);
  metadata->onnx.log_probs_output_name = ExtractJsonString(
      manifest, "log_probs", metadata->onnx.log_probs_output_name);

  metadata->vocab.tokens =
      ExtractJsonString(manifest, "tokens", metadata->vocab.tokens);
  metadata->vocab.words =
      ExtractJsonString(manifest, "words", metadata->vocab.words);
  metadata->vocab.blank_id =
      ExtractJsonInt(manifest, "blank_id", metadata->vocab.blank_id);
  metadata->vocab.sos_id =
      ExtractJsonInt(manifest, "sos_id", metadata->vocab.sos_id);
  metadata->vocab.eos_id =
      ExtractJsonInt(manifest, "eos_id", metadata->vocab.eos_id);

  metadata->decoder.type =
      ParseDecoderType(ExtractJsonString(manifest, "type",
                                         DecoderTypeName(metadata->decoder.type)));
  metadata->decoder.graph =
      ExtractJsonString(manifest, "graph", metadata->decoder.graph);
  metadata->decoder.beam =
      ExtractJsonFloat(manifest, "beam", metadata->decoder.beam);
  metadata->decoder.lattice_beam =
      ExtractJsonFloat(manifest, "lattice_beam", metadata->decoder.lattice_beam);
  metadata->decoder.max_active =
      ExtractJsonInt(manifest, "max_active", metadata->decoder.max_active);
  metadata->decoder.min_active =
      ExtractJsonInt(manifest, "min_active", metadata->decoder.min_active);
  metadata->decoder.acoustic_scale = ExtractJsonFloat(
      manifest, "acoustic_scale", metadata->decoder.acoustic_scale);
  metadata->decoder.lm_scale =
      ExtractJsonFloat(manifest, "lm_scale", metadata->decoder.lm_scale);
  metadata->decoder.length_penalty = ExtractJsonFloat(
      manifest, "length_penalty", metadata->decoder.length_penalty);
  metadata->decoder.blank_skip_thresh = ExtractJsonFloat(
      manifest, "blank_skip_thresh", metadata->decoder.blank_skip_thresh);
  metadata->decoder.nbest =
      ExtractJsonInt(manifest, "nbest", metadata->decoder.nbest);

  const auto yaml_path = model_dir / "config.yaml";
  if (std::filesystem::exists(yaml_path)) {
    std::string yaml;
    status = ReadTextFile(yaml_path, &yaml);
    if (!status.ok()) {
      return status;
    }
    ParseYamlConfig(yaml, metadata);
  }

  if (metadata->sample_rate <= 0 || metadata->feature_dim <= 0 ||
      metadata->frame_length_ms <= 0 || metadata->frame_shift_ms <= 0 ||
      metadata->streaming.chunk_size <= 0 || metadata->waveform_scale <= 0.0f) {
    return Status::InvalidArgument("model metadata contains non-positive "
                                   "sample rate, feature, frame, chunk "
                                   "dimension, or waveform scale");
  }
  return Status::OK();
}

Status ValidateModelPackageFiles(const ModelMetadata& metadata,
                                 bool require_runtime_models) {
  auto status = ValidateRequiredFile(metadata.model_dir / "manifest.json",
                                     "manifest.json");
  if (!status.ok()) {
    return status;
  }
  status = ValidateRequiredFile(metadata.Resolve(metadata.vocab.tokens),
                                "tokens.txt");
  if (!status.ok()) {
    return status;
  }
  status =
      ValidateRequiredFile(metadata.Resolve(metadata.vocab.words), "words.txt");
  if (!status.ok()) {
    return status;
  }
  status = ValidateRequiredFile(metadata.Resolve(metadata.decoder.graph),
                                "TLG.fst");
  if (!status.ok()) {
    return status;
  }

  SymbolTable tokens;
  status = tokens.Load(metadata.Resolve(metadata.vocab.tokens));
  if (!status.ok()) {
    return status;
  }
  SymbolTable words;
  status = words.Load(metadata.Resolve(metadata.vocab.words));
  if (!status.ok()) {
    return status;
  }
  if (!tokens.Contains(metadata.vocab.blank_id)) {
    return Status::InvalidArgument(
        "blank_id is not present in tokens.txt: " +
        std::to_string(metadata.vocab.blank_id));
  }
  Cmvn cmvn;
  status = cmvn.LoadIfPresent(metadata.Resolve("global_cmvn"),
                              metadata.feature_dim);
  if (!status.ok()) {
    return status;
  }
  if (metadata.decoder.type == DecoderType::kCtcWfst) {
    status = ValidateFstCompatibility(
        metadata.Resolve(metadata.decoder.graph), tokens.Size(), words.Size());
    if (!status.ok()) {
      return status;
    }
  }
  if (require_runtime_models) {
    status =
        ValidateRequiredFile(metadata.Resolve(metadata.onnx.encoder),
                             "encoder ONNX model");
    if (!status.ok()) {
      return status;
    }
    if (!metadata.onnx.ctc.empty()) {
      status = ValidateRequiredFile(metadata.Resolve(metadata.onnx.ctc),
                                    "CTC ONNX model");
      if (!status.ok()) {
        return status;
      }
    }
  }
  status = ValidateChecksumFileIfPresent(metadata.model_dir);
  if (!status.ok()) {
    return status;
  }
  return Status::OK();
}

std::string ModelSummary(const ModelMetadata& metadata) {
  std::ostringstream os;
  os << "model_dir: " << metadata.model_dir.string() << "\n"
     << "model_type: " << metadata.model_type << "\n"
     << "sample_rate: " << metadata.sample_rate << "\n"
     << "feature_dim: " << metadata.feature_dim << "\n"
     << "frame_length_ms: " << metadata.frame_length_ms << "\n"
     << "frame_shift_ms: " << metadata.frame_shift_ms << "\n"
     << "subsampling_rate: " << metadata.subsampling_rate << "\n"
     << "waveform_scale: " << metadata.waveform_scale << "\n"
     << "encoder: " << metadata.onnx.encoder << "\n"
     << "ctc: " << metadata.onnx.ctc << "\n"
     << "tokens: " << metadata.vocab.tokens << "\n"
     << "words: " << metadata.vocab.words << "\n"
     << "blank_id: " << metadata.vocab.blank_id << "\n"
     << "decoder: " << DecoderTypeName(metadata.decoder.type) << "\n"
     << "graph: " << metadata.decoder.graph << "\n"
     << "chunk_size: " << metadata.streaming.chunk_size << "\n"
     << "num_left_chunks: " << metadata.streaming.num_left_chunks << "\n";
  return os.str();
}

}  // namespace wenet_sdk::internal
