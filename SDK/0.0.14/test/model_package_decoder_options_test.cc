#include "package/model_package.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <unistd.h>

namespace {
bool Expect(bool value, const std::string& message) {
  if (!value) std::cerr << "FAIL: " << message << "\n";
  return value;
}
bool Near(double a, double b) { return std::fabs(a - b) < 1e-9; }
void Write(const std::filesystem::path& path, const std::string& text) {
  std::ofstream out(path);
  out << text;
}
}  // namespace

int main() {
  namespace fs = std::filesystem;
  const fs::path dir = fs::temp_directory_path() /
      ("asr_sdk_multi_lm_package_test_" + std::to_string(getpid()));
  fs::remove_all(dir);
  fs::create_directories(dir);
  Write(dir / "sdk_model.json", R"({
    "decoder_type":"flashlight_lexicon_kenlm",
    "model_path":"am.onnx", "tokens":"tokens.txt", "words":"words.txt",
    "lexicon":"lexicon.txt", "lm_search":"lm_search.json",
    "length_penalty":0.25, "beam_size":77, "smearing":"max"
  })");
  Write(dir / "lm_search.json", R"({
    "a.bin":{"type":"ngram","weight":0.5,"clip":true,
             "clip_lower":0.0,"clip_upper":8.0},
    "b.bin":{"type":"ngram","weight":0.8,"clip":false},
    "bias.bin":{"type":"bias","weight":1.5,
                "contact_lm_accumulation_factor":0.4,
                "slots":["<CONTACT>","<APP>"]}
  })");
  asr_sdk::EngineConfig config;
  config.model_dir = dir.string();
  auto package_or = asr_sdk::internal::LoadModelPackage(config);
  if (!Expect(package_or.ok(), package_or.status().ToString())) return 1;
  const auto& p = package_or.value();
  bool ok = true;
  ok &= Expect(p.fixed_lms.size() == 3, "three fixed LMs parsed");
  ok &= Expect(p.fixed_lms[0].filename == "a.bin", "filename order is stable");
  ok &= Expect(p.fixed_lms[0].clip && Near(p.fixed_lms[0].clip_upper, 8.0),
               "clip bounds parsed");
  ok &= Expect(!p.fixed_lms[1].clip, "unclipped ngram parsed");
  ok &= Expect(p.fixed_lms[2].type == asr_sdk::internal::FixedLmType::kBias &&
                   p.fixed_lms[2].slots.size() == 2,
               "multi-slot bias parsed");
  ok &= Expect(p.supports_runtime_slots, "bias enables runtime slots");
  ok &= Expect(Near(p.length_penalty, 0.25) &&
                   Near(p.flashlight_options.word_score, -0.25),
               "positive length penalty becomes negative word score");
  ok &= Expect(Near(p.flashlight_options.lm_weight, 1.0),
               "external LM weight fixed at one");

  Write(dir / "lm_search.json", R"({
    "bad.bin":{"type":"ngram","weight":1,"clip":true,
               "clip_lower":-1,"clip_upper":2}
  })");
  auto invalid = asr_sdk::internal::LoadModelPackage(config);
  ok &= Expect(!invalid.ok(), "negative clip lower rejected");

  Write(dir / "lm_search.json", R"({
    "bad.bin":{"type":"ngram","weight":1,"clip":false,
               "unexpected":true}
  })");
  invalid = asr_sdk::internal::LoadModelPackage(config);
  ok &= Expect(!invalid.ok(), "unknown LM parameter rejected");

  Write(dir / "lm_search.json", R"({
    "bad.bin":{"type":"ngram","weight":1,"weight":2,"clip":false}
  })");
  invalid = asr_sdk::internal::LoadModelPackage(config);
  ok &= Expect(!invalid.ok(), "duplicate JSON key rejected");

  Write(dir / "lm_search.json", R"({
    "bad.bin":{"type":"ngram","weight":0,"clip":false}
  })");
  invalid = asr_sdk::internal::LoadModelPackage(config);
  ok &= Expect(!invalid.ok(), "non-positive LM weight rejected");

  Write(dir / "lm_search.json", R"({
    "a.bin":{"type":"ngram","weight":1,"clip":false}
  })");
  Write(dir / "sdk_model.json", R"({
    "decoder_type":"flashlight_lexicon_kenlm",
    "model_path":"am.onnx", "tokens":"tokens.txt", "words":"words.txt",
    "lexicon":"lexicon.txt", "lm_search":"lm_search.json",
    "itn_language":"en", "itn_tagger":"en_itn_tagger.fst",
    "itn_verbalizer":"en_itn_verbalizer.fst"
  })");
  auto itn_package = asr_sdk::internal::LoadModelPackage(config);
  ok &= Expect(itn_package.ok() && itn_package.value().has_itn,
               "all English ITN fields enable ITN");
  Write(dir / "sdk_model.json", R"({
    "decoder_type":"flashlight_lexicon_kenlm",
    "model_path":"am.onnx", "tokens":"tokens.txt", "words":"words.txt",
    "lexicon":"lexicon.txt", "lm_search":"lm_search.json",
    "itn_language":"en", "itn_tagger":"en_itn_tagger.fst"
  })");
  invalid = asr_sdk::internal::LoadModelPackage(config);
  ok &= Expect(!invalid.ok(), "partial ITN configuration rejected");
  fs::remove_all(dir);
  return ok ? 0 : 1;
}
