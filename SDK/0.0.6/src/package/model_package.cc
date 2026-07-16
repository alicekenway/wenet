#include "package/model_package.h"

#include <algorithm>

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
    package.flashlight_options.lm_weight =
        FindJsonDoubleValue(manifest_json, "lm_weight",
                            package.flashlight_options.lm_weight);
    package.flashlight_options.word_score =
        FindJsonDoubleValue(manifest_json, "word_score",
                            package.flashlight_options.word_score);
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
    package.contact_class_word_declared =
        HasJsonKey(manifest_json, "contact_class_word");
    if (package.contact_class_word_declared) {
      package.contact_class_word =
          FindJsonStringValue(manifest_json, "contact_class_word", "");
    }
    package.contact_lm_declared = HasJsonKey(manifest_json, "contact_lm");
    if (package.contact_lm_declared) {
      const std::string contact_lm =
          FindJsonStringValue(manifest_json, "contact_lm", "");
      if (!contact_lm.empty()) {
        package.contact_kenlm_bin = ResolveUnder(package.root, contact_lm);
      }
    }
    package.contact_lm_weight_declared =
        HasJsonKey(manifest_json, "contact_lm_weight");
    if (package.contact_lm_weight_declared) {
      package.contact_lm_weight = FindJsonDoubleValue(
          manifest_json, "contact_lm_weight", package.contact_lm_weight);
    }
    package.contact_lm_mode_declared =
        HasJsonKey(manifest_json, "contact_lm_mode");
    if (package.contact_lm_mode_declared) {
      package.contact_lm_mode =
          FindJsonStringValue(manifest_json, "contact_lm_mode", "");
    }
    package.contact_lm_metadata_declared =
        HasJsonKey(manifest_json, "contact_lm_metadata");
    if (package.contact_lm_metadata_declared) {
      const std::string metadata =
          FindJsonStringValue(manifest_json, "contact_lm_metadata", "");
      if (!metadata.empty()) {
        package.contact_lm_metadata = ResolveUnder(package.root, metadata);
      }
    }
    package.contact_lm_accumulation_factor_declared =
        HasJsonKey(manifest_json, "contact_lm_accumulation_factor");
    if (package.contact_lm_accumulation_factor_declared) {
      package.contact_lm_accumulation_factor = FindJsonDoubleValue(
          manifest_json, "contact_lm_accumulation_factor",
          package.contact_lm_accumulation_factor);
    }
    if (!package.contact_lm_metadata.empty() &&
        FileExists(package.contact_lm_metadata)) {
      std::string metadata_json;
      Status metadata_status =
          ReadTextFile(package.contact_lm_metadata, &metadata_json);
      if (!metadata_status.ok()) {
        return metadata_status;
      }
      package.contact_lm_metadata_format =
          FindJsonStringValue(metadata_json, "format", "");
      package.contact_lm_max_bonus =
          FindJsonDoubleValue(metadata_json, "max_bonus", 0.0);
    }

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
    package.kenlm_bin = ResolveUnder(
        package.root, FindJsonStringValue(manifest_json, "lm", "lm.bin"));
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
    package.kenlm_bin = package.root / "lm.bin";
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
        FileExists(package.lexicon_txt) && FileExists(package.kenlm_bin)) {
      package.decoder_type = "flashlight_lexicon_kenlm";
    }
  }

  package.encoder_onnx = package.runtime_dir / "encoder.onnx";
  package.ctc_onnx = package.runtime_dir / "ctc.onnx";
  package.decoder_onnx = package.runtime_dir / "decoder.onnx";
  package.flashlight_options.nbest = package.nbest;
  package.has_flashlight_decoder =
      package.decoder_type == "flashlight_lexicon_kenlm";
  package.supports_runtime_contacts =
      package.has_flashlight_decoder && package.contact_class_word_declared &&
      !package.contact_class_word.empty() && package.contact_lm_declared &&
      !package.contact_kenlm_bin.empty() &&
      package.contact_lm_mode == "pattern_bias_v1" &&
      !package.contact_lm_metadata.empty();
  package.has_wfst = !package.has_flashlight_decoder && FileExists(package.tlg_fst);
  return package;
}

}  // namespace asr_sdk::internal
