#include "package/model_package.h"

#include <algorithm>
#include <cmath>
#include <set>

#include "utils/file_utils.h"
#include "utils/json.h"

namespace asr_sdk::internal {
namespace {

std::string NonEmptyOr(std::string value, const std::string& fallback) {
  return value.empty() ? fallback : std::move(value);
}

int PositiveOr(int value, int fallback) {
  return value > 0 ? value : fallback;
}

bool HasJsonKey(const std::string& json, const std::string& key) {
  return json.find("\"" + key + "\"") != std::string::npos;
}

Status InvalidLmField(const std::string& file, const std::string& message) {
  return Status::InvalidArgument("lm_search entry '" + file + "': " + message);
}

StatusOr<std::vector<FixedLmConfig>> LoadFixedLms(
    const std::filesystem::path& root,
    const std::filesystem::path& config_path) {
  std::string text;
  Status read = ReadTextFile(config_path, &text);
  if (!read.ok()) return read;
  auto parsed = ParseJson(text);
  if (!parsed.ok()) return parsed.status();
  if (!parsed.value().IsObject()) {
    return Status::InvalidArgument("lm_search JSON must be a top-level object");
  }
  std::vector<FixedLmConfig> configs;
  for (const auto& [filename, value] : parsed.value().AsObject()) {
    const std::filesystem::path relative(filename);
    if (filename.empty() || relative.is_absolute() || relative.has_parent_path() ||
        relative.extension() != ".bin") {
      return InvalidLmField(filename,
                            "key must be a .bin basename in the package root");
    }
    if (!value.IsObject()) {
      return InvalidLmField(filename, "parameters must be an object");
    }
    const auto& object = value.AsObject();
    auto require = [&](const char* key) -> StatusOr<const JsonValue*> {
      const auto it = object.find(key);
      if (it == object.end()) {
        return InvalidLmField(filename, std::string("missing '") + key + "'");
      }
      return &it->second;
    };
    auto type_value = require("type");
    auto weight_value = require("weight");
    if (!type_value.ok()) return type_value.status();
    if (!weight_value.ok()) return weight_value.status();
    if (!type_value.value()->IsString()) {
      return InvalidLmField(filename, "'type' must be a string");
    }
    if (!weight_value.value()->IsNumber() ||
        !std::isfinite(weight_value.value()->AsNumber()) ||
        weight_value.value()->AsNumber() <= 0.0) {
      return InvalidLmField(filename, "'weight' must be finite and > 0");
    }
    FixedLmConfig config;
    config.filename = filename;
    config.path = root / relative;
    config.weight = weight_value.value()->AsNumber();
    const std::string type = type_value.value()->AsString();
    std::set<std::string> allowed{"type", "weight"};
    if (type == "ngram") {
      config.type = FixedLmType::kNgram;
      allowed.insert("clip");
      allowed.insert("clip_lower");
      allowed.insert("clip_upper");
      auto clip = require("clip");
      if (!clip.ok()) return clip.status();
      if (!clip.value()->IsBool()) {
        return InvalidLmField(filename, "'clip' must be boolean");
      }
      config.clip = clip.value()->AsBool();
      const auto lower = object.find("clip_lower");
      const auto upper = object.find("clip_upper");
      if (config.clip) {
        if (lower == object.end() || upper == object.end() ||
            !lower->second.IsNumber() || !upper->second.IsNumber()) {
          return InvalidLmField(
              filename, "clipping requires numeric clip_lower and clip_upper");
        }
        config.clip_lower = lower->second.AsNumber();
        config.clip_upper = upper->second.AsNumber();
        if (!std::isfinite(config.clip_lower) ||
            !std::isfinite(config.clip_upper) || config.clip_lower < 0.0 ||
            config.clip_lower > config.clip_upper) {
          return InvalidLmField(
              filename, "clip bounds must satisfy 0 <= lower <= upper");
        }
      } else if (lower != object.end() || upper != object.end()) {
        return InvalidLmField(filename,
                              "clip bounds are forbidden when clip is false");
      }
    } else if (type == "bias") {
      config.type = FixedLmType::kBias;
      allowed.insert("contact_lm_accumulation_factor");
      allowed.insert("slots");
      auto factor = require("contact_lm_accumulation_factor");
      auto slots = require("slots");
      if (!factor.ok()) return factor.status();
      if (!slots.ok()) return slots.status();
      if (!factor.value()->IsNumber() ||
          !std::isfinite(factor.value()->AsNumber()) ||
          factor.value()->AsNumber() < 0.0 ||
          factor.value()->AsNumber() > 1.0) {
        return InvalidLmField(filename,
                              "accumulation factor must be in [0, 1]");
      }
      config.accumulation_factor = factor.value()->AsNumber();
      if (!slots.value()->IsArray() || slots.value()->AsArray().empty()) {
        return InvalidLmField(filename, "'slots' must be a non-empty array");
      }
      std::set<std::string> unique;
      for (const JsonValue& slot : slots.value()->AsArray()) {
        if (!slot.IsString() || slot.AsString().size() < 3 ||
            slot.AsString().front() != '<' || slot.AsString().back() != '>' ||
            slot.AsString().find_first_of(" \t\r\n") != std::string::npos) {
          return InvalidLmField(filename,
                                "slot tokens must have form <SLOT>");
        }
        if (!unique.insert(slot.AsString()).second) {
          return InvalidLmField(filename, "duplicate slot token: " +
                                              slot.AsString());
        }
        config.slots.push_back(slot.AsString());
      }
    } else {
      return InvalidLmField(filename, "unknown type '" + type + "'");
    }
    for (const auto& [key, ignored] : object) {
      if (allowed.find(key) == allowed.end()) {
        return InvalidLmField(filename, "unknown field '" + key + "'");
      }
    }
    configs.push_back(std::move(config));
  }
  if (configs.empty()) {
    return Status::InvalidArgument("lm_search JSON must contain at least one LM");
  }
  return configs;
}

}  // namespace

StatusOr<ModelPackage> LoadModelPackage(const EngineConfig& config) {
  if (config.model_dir.empty()) {
    return Status::InvalidArgument("model_dir is empty");
  }
  ModelPackage package;
  package.root = std::filesystem::absolute(config.model_dir);
  package.manifest = package.root / "sdk_model.json";
  package.runtime_dir = package.root;
  package.sample_rate = PositiveOr(config.sample_rate, 16000);
  package.chunk_size = PositiveOr(config.chunk_size, 16);
  package.num_left_chunks = config.num_left_chunks;
  package.nbest = std::clamp(config.nbest, 1, 10);
  package.enable_continuous_decoding = config.enable_continuous_decoding;
  package.enable_timestamps = config.enable_timestamps;
  package.language = config.language.empty() ? "chs" : config.language;

  if (!DirectoryExists(package.root)) {
    return Status::NotFound("model_dir does not exist: " +
                            package.root.string());
  }

  std::string manifest_json;
  if (FileExists(package.manifest)) {
    package.has_manifest = true;
    Status read = ReadTextFile(package.manifest, &manifest_json);
    if (!read.ok()) {
      return read;
    }
    const std::string onnx_dir =
        FindJsonStringValue(manifest_json, "onnx_dir", ".");
    package.runtime_dir = ResolveUnder(package.root, onnx_dir);
    package.sample_rate =
        PositiveOr(FindJsonIntValue(manifest_json, "sample_rate", 16000),
                   package.sample_rate);
    package.chunk_size =
        PositiveOr(FindJsonIntValue(manifest_json, "chunk_size", 16),
                   package.chunk_size);
    package.num_left_chunks =
        FindJsonIntValue(manifest_json, "num_left_chunks",
                         package.num_left_chunks);
    package.nbest = std::clamp(
        PositiveOr(FindJsonIntValue(manifest_json, "nbest", package.nbest),
                   package.nbest),
        1, 10);
    package.flashlight_options.beam_size = PositiveOr(
        FindJsonIntValue(manifest_json, "beam_size",
                         package.flashlight_options.beam_size),
        package.flashlight_options.beam_size);
    package.flashlight_options.beam_size_token = PositiveOr(
        FindJsonIntValue(manifest_json, "beam_size_token",
                         package.flashlight_options.beam_size_token),
        package.flashlight_options.beam_size_token);
    package.flashlight_options.beam_threshold =
        FindJsonDoubleValue(manifest_json, "beam_threshold",
                            package.flashlight_options.beam_threshold);
    package.flashlight_options.lm_weight = 1.0;
    package.length_penalty =
        FindJsonDoubleValue(manifest_json, "length_penalty", 0.0);
    package.flashlight_options.word_score = -package.length_penalty;
    package.flashlight_options.unk_score =
        FindJsonDoubleValue(manifest_json, "unk_score",
                            package.flashlight_options.unk_score);
    package.flashlight_options.sil_score =
        FindJsonDoubleValue(manifest_json, "sil_score",
                            package.flashlight_options.sil_score);
    package.flashlight_options.log_add =
        FindJsonBoolValue(manifest_json, "log_add",
                          package.flashlight_options.log_add);
    package.flashlight_options.allow_unk =
        FindJsonBoolValue(manifest_json, "allow_unk",
                          package.flashlight_options.allow_unk);
    package.flashlight_options.smearing =
        NonEmptyOr(FindJsonStringValue(manifest_json, "smearing", ""),
                   package.flashlight_options.smearing);
    package.debug = FindJsonBoolValue(manifest_json, "debug", false);
    package.enable_continuous_decoding =
        FindJsonBoolValue(manifest_json, "enable_continuous_decoding",
                          package.enable_continuous_decoding);
    package.enable_timestamps =
        FindJsonBoolValue(manifest_json, "enable_timestamp",
                          package.enable_timestamps);
    package.language =
        NonEmptyOr(FindJsonStringValue(manifest_json, "language_type", ""),
                   package.language);
    package.decoder_type =
        NonEmptyOr(FindJsonStringValue(manifest_json, "decoder_type", ""),
                   package.decoder_type);
    package.feature_type =
        NonEmptyOr(FindJsonStringValue(manifest_json, "feature_type", ""),
                   package.feature_type);
    package.blank_token =
        NonEmptyOr(FindJsonStringValue(manifest_json, "blank_token", ""),
                   package.blank_token);
    package.sil_token =
        NonEmptyOr(FindJsonStringValue(manifest_json, "sil_token", ""),
                   package.sil_token);
    package.unk_word =
        NonEmptyOr(FindJsonStringValue(manifest_json, "unk_word", ""),
                   package.unk_word);
    package.units_txt = ResolveUnder(
        package.root, FindJsonStringValue(manifest_json, "unit_path", "units.txt"));
    package.tokens_txt = ResolveUnder(
        package.root,
        FindJsonStringValue(manifest_json, "tokens", "tokens.txt"));
    package.words_txt = ResolveUnder(
        package.root,
        FindJsonStringValue(manifest_json, "words",
                            FindJsonStringValue(manifest_json, "dict_path",
                                                "words.txt")));
    package.sherpa_ctc_onnx = ResolveUnder(
        package.root,
        FindJsonStringValue(manifest_json, "model_path", "model.onnx"));
    package.lexicon_txt = ResolveUnder(
        package.root,
        FindJsonStringValue(manifest_json, "lexicon", "lexicon.txt"));
    const std::string lm_search =
        FindJsonStringValue(manifest_json, "lm_search", "");
    if (!lm_search.empty()) {
      package.lm_search_json = ResolveUnder(package.root, lm_search);
      auto fixed_lms = LoadFixedLms(package.root, package.lm_search_json);
      if (!fixed_lms.ok()) return fixed_lms.status();
      package.fixed_lms = std::move(fixed_lms).value();
    }
    const bool has_mapping_key = HasJsonKey(manifest_json, "mapping");
    const std::string mapping = FindJsonStringValue(
        manifest_json, "mapping", has_mapping_key ? "" : "output_mapping.txt");
    if (!mapping.empty()) {
      const auto mapping_path = ResolveUnder(package.root, mapping);
      if (has_mapping_key || FileExists(mapping_path)) {
        package.output_mapping_txt = mapping_path;
      }
    }
    const std::string final_mapping =
        FindJsonStringValue(manifest_json, "final_mapping", "");
    package.final_output_mapping_txt =
        final_mapping.empty() ? std::filesystem::path()
                              : ResolveUnder(package.root, final_mapping);
    package.tlg_fst = ResolveUnder(
        package.root, FindJsonStringValue(manifest_json, "fst_path", "TLG.fst"));
  } else {
    package.units_txt = package.root / "units.txt";
    package.tokens_txt = package.root / "tokens.txt";
    package.words_txt = package.root / "words.txt";
    package.sherpa_ctc_onnx = package.root / "model.onnx";
    package.lexicon_txt = package.root / "lexicon.txt";
    const auto mapping_path = package.root / "output_mapping.txt";
    if (FileExists(mapping_path)) {
      package.output_mapping_txt = mapping_path;
    }
    const auto final_mapping_path = package.root / "final_output_mapping.txt";
    if (FileExists(final_mapping_path)) {
      package.final_output_mapping_txt = final_mapping_path;
    }
    package.tlg_fst = package.root / "TLG.fst";
    if (FileExists(package.sherpa_ctc_onnx) && FileExists(package.tokens_txt) &&
        FileExists(package.lexicon_txt) && FileExists(package.lm_search_json)) {
      package.decoder_type = "flashlight_lexicon_kenlm";
    }
  }

  package.encoder_onnx = package.runtime_dir / "encoder.onnx";
  package.ctc_onnx = package.runtime_dir / "ctc.onnx";
  package.decoder_onnx = package.runtime_dir / "decoder.onnx";
  package.flashlight_options.nbest = package.nbest;
  package.has_flashlight_decoder =
      package.decoder_type == "flashlight_lexicon_kenlm";
  package.supports_runtime_slots = package.has_flashlight_decoder &&
      std::any_of(package.fixed_lms.begin(), package.fixed_lms.end(),
                  [](const FixedLmConfig& lm) {
                    return lm.type == FixedLmType::kBias;
                  });
  package.has_wfst = !package.has_flashlight_decoder && FileExists(package.tlg_fst);
  return package;
}

}  // namespace asr_sdk::internal
