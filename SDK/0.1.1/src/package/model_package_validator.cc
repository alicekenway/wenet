#include "package/model_package_validator.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <exception>
#include <fstream>

#include "flashlight_decoder/lexicon_loader.h"
#include "flashlight_decoder/output_sequence_mapper.h"
#include "flashlight_decoder/word_dictionary.h"
#include "sherpa_onnx_wenet/token_table.h"
#include "utils/file_utils.h"

namespace asr_sdk::internal {
namespace {

void AddFileLine(const char* label, const std::filesystem::path& path,
                 ModelPackageReport* report) {
  const bool ok = FileExists(path);
  report->lines.push_back(std::string(label) + ": " +
                          (ok ? "ok " : "missing ") + path.string());
  report->ok = report->ok && ok;
}

void AddOptionalMappingLine(const char* label, const std::filesystem::path& path,
                            ModelPackageReport* report) {
  if (!path.empty() && FileExists(path)) {
    AddFileLine(label, path, report);
  } else {
    report->lines.push_back(std::string(label) + ": identity");
  }
}

Status RequireFile(const std::filesystem::path& path, const char* label) {
  if (!FileExists(path)) {
    return Status::NotFound(std::string(label) + " not found: " +
                            path.string());
  }
  return Status::Ok();
}

int CountNonEmptyLines(const std::filesystem::path& path) {
  std::ifstream in(path);
  int count = 0;
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty()) {
      ++count;
    }
  }
  return count;
}

std::string LowerAscii(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return value;
}

int MappingRuleCount(const std::filesystem::path& path,
                     const flashlight_decoder::WordDictionary& words) {
  if (path.empty() || !FileExists(path)) {
    return 0;
  }
  return flashlight_decoder::OutputSequenceMapper::Load(path, words).RuleCount();
}

Status ValidateFlashlightOptions(
    const flashlight_decoder::FlashlightDecoderOptions& options) {
  if (options.beam_size <= 0) {
    return Status::InvalidArgument("decoder beam_size must be > 0");
  }
  if (options.beam_size_token <= 0) {
    return Status::InvalidArgument("decoder beam_size_token must be > 0");
  }
  if (!std::isfinite(options.beam_threshold) ||
      options.beam_threshold < 0.0) {
    return Status::InvalidArgument(
        "decoder beam_threshold must be finite and >= 0");
  }
  if (!std::isfinite(options.lm_weight)) {
    return Status::InvalidArgument("decoder lm_weight must be finite");
  }
  if (!std::isfinite(options.word_score)) {
    return Status::InvalidArgument("decoder word_score must be finite");
  }
  if (!std::isfinite(options.unk_score)) {
    return Status::InvalidArgument("decoder unk_score must be finite");
  }
  if (!std::isfinite(options.sil_score)) {
    return Status::InvalidArgument("decoder sil_score must be finite");
  }
  if (options.nbest < 1 || options.nbest > 10) {
    return Status::InvalidArgument("decoder nbest must be between 1 and 10");
  }
  const std::string smearing = LowerAscii(options.smearing);
  if (smearing != "none" && smearing != "max" && smearing != "logadd" &&
      smearing != "log_add") {
    return Status::InvalidArgument(
        "decoder smearing must be one of: none, max, logadd, log_add");
  }
  return Status::Ok();
}

Status ValidateFlashlightPackage(const ModelPackage& package) {
  Status options_status = ValidateFlashlightOptions(package.flashlight_options);
  if (!options_status.ok()) {
    return options_status;
  }
  for (const auto& item :
       {std::pair{package.sherpa_ctc_onnx, "model.onnx"},
        std::pair{package.tokens_txt, "tokens.txt"},
        std::pair{package.words_txt, "words.txt"},
        std::pair{package.lexicon_txt, "lexicon.txt"},
        std::pair{package.kenlm_bin, "lm.bin"}}) {
    Status status = RequireFile(item.first, item.second);
    if (!status.ok()) {
      return status;
    }
  }
  try {
    sherpa_onnx_wenet::TokenTable tokens(package.tokens_txt);
    if (!tokens.Contains(package.blank_token)) {
      return Status::InvalidArgument("blank token not found in tokens.txt: " +
                                     package.blank_token);
    }
    if (!tokens.Contains(package.sil_token)) {
      return Status::InvalidArgument("sil token not found in tokens.txt: " +
                                     package.sil_token);
    }
    flashlight_decoder::WordDictionary words(package.words_txt);
    if (!words.Contains(package.unk_word)) {
      return Status::InvalidArgument("unknown word not found in words.txt: " +
                                     package.unk_word);
    }
    (void)flashlight_decoder::LoadLexicon(package.lexicon_txt, words, tokens);
	    if (!package.output_mapping_txt.empty() &&
	        FileExists(package.output_mapping_txt)) {
	      (void)flashlight_decoder::OutputSequenceMapper::Load(
	          package.output_mapping_txt, words);
	    }
    if (!package.final_output_mapping_txt.empty() &&
        FileExists(package.final_output_mapping_txt)) {
      (void)flashlight_decoder::OutputSequenceMapper::Load(
          package.final_output_mapping_txt, words);
    }
	  } catch (const std::exception& e) {
    return Status::InvalidArgument(e.what());
  }
  return Status::Ok();
}

}  // namespace

Status ValidateModelPackage(const ModelPackage& package) {
  if (!DirectoryExists(package.root)) {
    return Status::NotFound("model root not found: " + package.root.string());
  }
  if (!DirectoryExists(package.runtime_dir)) {
    return Status::NotFound("ONNX runtime directory not found: " +
                            package.runtime_dir.string());
  }
  if (package.has_flashlight_decoder) {
    if (package.sample_rate != 16000) {
      return Status::InvalidArgument(
          "only 16 kHz PCM is supported by this Flashlight CTC package");
    }
    return ValidateFlashlightPackage(package);
  }
  for (const auto& item :
       {std::pair{package.encoder_onnx, "encoder.onnx"},
        std::pair{package.ctc_onnx, "ctc.onnx"},
        std::pair{package.decoder_onnx, "decoder.onnx"},
        std::pair{package.units_txt, "units.txt"}}) {
    Status status = RequireFile(item.first, item.second);
    if (!status.ok()) {
      return status;
    }
  }
  if (package.has_wfst || FileExists(package.tlg_fst)) {
    Status status = RequireFile(package.tlg_fst, "TLG.fst");
    if (!status.ok()) {
      return status;
    }
    status = RequireFile(package.words_txt, "words.txt");
    if (!status.ok()) {
      return status;
    }
  }
  if (package.sample_rate != 16000) {
    return Status::InvalidArgument(
        "only 16 kHz PCM is supported by this WeNet runtime package");
  }
  if (package.chunk_size == 0) {
    return Status::InvalidArgument("chunk_size must not be 0");
  }
  return Status::Ok();
}

ModelPackageReport InspectModelPackage(const ModelPackage& package) {
  ModelPackageReport report;
  report.ok = true;
  report.lines.push_back("model_dir: " + package.root.string());
  report.lines.push_back("manifest: " +
                         std::string(package.has_manifest ? "yes " : "no ") +
                         package.manifest.string());
  report.lines.push_back("runtime_dir: " + package.runtime_dir.string());
  report.lines.push_back("sample_rate: " + std::to_string(package.sample_rate));
  report.lines.push_back("chunk_size: " + std::to_string(package.chunk_size));
  report.lines.push_back("num_left_chunks: " +
                         std::to_string(package.num_left_chunks));
  report.lines.push_back("nbest: " + std::to_string(package.nbest));
  report.lines.push_back("decoder_type: " + package.decoder_type);
  if (package.has_flashlight_decoder) {
    AddFileLine("model.onnx", package.sherpa_ctc_onnx, &report);
    AddFileLine("tokens.txt", package.tokens_txt, &report);
    AddFileLine("words.txt", package.words_txt, &report);
	    AddFileLine("lexicon.txt", package.lexicon_txt, &report);
	    AddFileLine("lm.bin", package.kenlm_bin, &report);
    AddOptionalMappingLine("am_mapping", package.output_mapping_txt, &report);
    AddOptionalMappingLine("final_mapping", package.final_output_mapping_txt,
                           &report);
	    try {
	      sherpa_onnx_wenet::TokenTable tokens(package.tokens_txt);
	      flashlight_decoder::WordDictionary words(package.words_txt);
	      const auto lexicon =
	          flashlight_decoder::LoadLexicon(package.lexicon_txt, words, tokens);
      const int am_mapping_rules =
          MappingRuleCount(package.output_mapping_txt, words);
      const int final_mapping_rules =
          MappingRuleCount(package.final_output_mapping_txt, words);
	      report.lines.push_back("token count: " +
	                             std::to_string(tokens.ModelVocabSize()));
      report.lines.push_back("word count: " + std::to_string(words.Size()));
      report.lines.push_back("lexicon entry count: " +
                             std::to_string(lexicon.size()));
      report.lines.push_back("am_mapping rule count: " +
                             std::to_string(am_mapping_rules));
      report.lines.push_back("final_mapping rule count: " +
                             std::to_string(final_mapping_rules));
	      report.lines.push_back("blank token/id: " + package.blank_token + "/" +
                             std::to_string(tokens.Id(package.blank_token)));
      report.lines.push_back("sil token/id: " + package.sil_token + "/" +
                             std::to_string(tokens.Id(package.sil_token)));
      report.lines.push_back("unk word/id: " + package.unk_word + "/" +
                             std::to_string(words.Id(package.unk_word)));
      const auto& options = package.flashlight_options;
      report.lines.push_back("decoder beam_size: " +
                             std::to_string(options.beam_size));
      report.lines.push_back("decoder beam_size_token: " +
                             std::to_string(options.beam_size_token));
      report.lines.push_back("decoder beam_threshold: " +
                             std::to_string(options.beam_threshold));
      report.lines.push_back(std::string("debug: ") +
                             (package.debug ? "true" : "false"));
	      report.lines.push_back("decoder lm_weight final KenLM rescore weight: " +
	                             std::to_string(options.lm_weight));
	      report.lines.push_back("decoder word_score final insertion score: " +
	                             std::to_string(options.word_score));
	      report.lines.push_back("decoder unk_score final unk penalty: " +
	                             std::to_string(options.unk_score));
      report.lines.push_back("decoder sil_score: " +
                             std::to_string(options.sil_score));
      report.lines.push_back(std::string("decoder log_add: ") +
                             (options.log_add ? "true" : "false"));
      report.lines.push_back(std::string("decoder allow_unk: ") +
                             (options.allow_unk ? "true" : "false"));
      report.lines.push_back("decoder smearing: " + options.smearing);
    } catch (const std::exception& e) {
      report.ok = false;
      report.lines.push_back(std::string("resource validation: ") + e.what());
    }
    const Status status = ValidateModelPackage(package);
    if (!status.ok()) {
      report.ok = false;
      report.lines.push_back("status: " + status.ToString());
    } else {
      report.lines.push_back("status: OK");
    }
    return report;
  }
  AddFileLine("encoder.onnx", package.encoder_onnx, &report);
  AddFileLine("ctc.onnx", package.ctc_onnx, &report);
  AddFileLine("decoder.onnx", package.decoder_onnx, &report);
  AddFileLine("units.txt", package.units_txt, &report);
  if (FileExists(package.tlg_fst)) {
    AddFileLine("TLG.fst", package.tlg_fst, &report);
    AddFileLine("words.txt", package.words_txt, &report);
  } else {
    report.lines.push_back("TLG.fst: not configured");
  }
  const Status status = ValidateModelPackage(package);
  if (!status.ok()) {
    report.ok = false;
    report.lines.push_back("status: " + status.ToString());
  } else {
    report.lines.push_back("status: OK");
  }
  return report;
}

}  // namespace asr_sdk::internal
