#include "package/model_package.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

namespace {

bool Expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << "\n";
    return false;
  }
  return true;
}

bool ExpectNear(double actual, double expected, const std::string& message) {
  return Expect(std::fabs(actual - expected) < 1e-9,
                message + " actual=" + std::to_string(actual) +
                    " expected=" + std::to_string(expected));
}

}  // namespace

int main() {
  namespace fs = std::filesystem;
  const fs::path dir =
      fs::temp_directory_path() /
      ("asr_sdk_model_package_options_test_" + std::to_string(getpid()));
  fs::remove_all(dir);
  fs::create_directories(dir);

  std::ofstream manifest(dir / "sdk_model.json");
  manifest << R"json({
  "decoder_type": "flashlight_lexicon_kenlm",
  "model_path": "am.onnx",
  "tokens": "am.tokens",
  "words": "out.words",
  "lexicon": "out.lexicon",
  "lm": "out.lm",
  "mapping": "",
  "final_mapping": "final_output_mapping.txt",
  "debug": true,
  "blank_token": "<blank>",
  "sil_token": "|",
  "unk_word": "<unknown>",
  "sample_rate": 16000,
  "beam_size": 77,
  "beam_size_token": 33,
  "beam_threshold": 19.5,
  "lm_weight": 0.35,
  "word_score": 0.25,
  "unk_score": -9.75,
  "sil_score": -0.125,
  "log_add": true,
  "allow_unk": false,
  "smearing": "logadd",
  "nbest": 4
})json";
  manifest.close();

  asr_sdk::EngineConfig config;
  config.model_dir = dir.string();
  config.nbest = 2;
  auto package_or = asr_sdk::internal::LoadModelPackage(config);
  if (!Expect(package_or.ok(), package_or.status().ToString())) {
    return 1;
  }
  const auto& package = package_or.value();
  const auto& options = package.flashlight_options;

	  bool ok = true;
	  ok &= Expect(package.has_flashlight_decoder, "Flashlight decoder selected");
	  ok &= Expect(package.output_mapping_txt.empty(), "empty mapping disables mapping path");
  ok &= Expect(package.final_output_mapping_txt == dir / "final_output_mapping.txt",
               "final mapping path parsed");
  ok &= Expect(package.debug, "debug parsed");
	  ok &= Expect(package.blank_token == "<blank>", "blank token parsed");
  ok &= Expect(package.sil_token == "|", "sil token parsed");
  ok &= Expect(package.unk_word == "<unknown>", "unknown word parsed");
  ok &= Expect(package.nbest == 4, "package nbest parsed");
  ok &= Expect(options.nbest == 4, "decoder nbest follows package nbest");
  ok &= Expect(options.beam_size == 77, "beam_size parsed");
  ok &= Expect(options.beam_size_token == 33, "beam_size_token parsed");
  ok &= ExpectNear(options.beam_threshold, 19.5, "beam_threshold parsed");
  ok &= ExpectNear(options.lm_weight, 0.35, "lm_weight parsed");
  ok &= ExpectNear(options.word_score, 0.25, "word_score parsed");
  ok &= ExpectNear(options.unk_score, -9.75, "unk_score parsed");
  ok &= ExpectNear(options.sil_score, -0.125, "sil_score parsed");
  ok &= Expect(options.log_add, "log_add parsed");
  ok &= Expect(!options.allow_unk, "allow_unk parsed");
  ok &= Expect(options.smearing == "logadd", "smearing parsed");

  fs::remove_all(dir);
  return ok ? 0 : 1;
}
