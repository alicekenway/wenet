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
    package.enable_continuous_decoding =
        FindJsonBoolValue(manifest_json, "enable_continuous_decoding",
                          package.enable_continuous_decoding);
    package.enable_timestamps =
        FindJsonBoolValue(manifest_json, "enable_timestamp",
                          package.enable_timestamps);
    package.language =
        NonEmptyOr(FindJsonStringValue(manifest_json, "language_type", ""),
                   package.language);

    package.units_txt = ResolveUnder(
        package.root, FindJsonStringValue(manifest_json, "unit_path", "units.txt"));
    package.words_txt = ResolveUnder(
        package.root, FindJsonStringValue(manifest_json, "dict_path", "words.txt"));
    package.tlg_fst = ResolveUnder(
        package.root, FindJsonStringValue(manifest_json, "fst_path", "TLG.fst"));
  } else {
    package.units_txt = package.root / "units.txt";
    package.words_txt = package.root / "words.txt";
    package.tlg_fst = package.root / "TLG.fst";
  }

  package.encoder_onnx = package.runtime_dir / "encoder.onnx";
  package.ctc_onnx = package.runtime_dir / "ctc.onnx";
  package.decoder_onnx = package.runtime_dir / "decoder.onnx";
  package.has_wfst = FileExists(package.tlg_fst);
  return package;
}

}  // namespace asr_sdk::internal
