#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight_decoder/max_fusion_lm.h"
#include "flashlight_decoder/output_sequence_mapper.h"
#include "flashlight_decoder/word_dictionary.h"
#include "package/model_package.h"

namespace {
namespace fs = std::filesystem;
using asr_sdk::internal::FixedLmConfig;
using asr_sdk::internal::FixedLmType;
using asr_sdk::internal::flashlight_decoder::MaxFusionLm;
using asr_sdk::internal::flashlight_decoder::OutputSequenceMapper;
using asr_sdk::internal::flashlight_decoder::WordDictionary;

void Write(const fs::path& path, const std::string& text) {
  std::ofstream out(path);
  if (!out) throw std::runtime_error("failed to write " + path.string());
  out << text;
}

void Near(float actual, float expected, const std::string& message) {
  if (std::fabs(actual - expected) > 1e-4f) {
    throw std::runtime_error(message + " actual=" + std::to_string(actual) +
                             " expected=" + std::to_string(expected));
  }
}

std::string Arpa() {
  return "\\data\\\n"
         "ngram 1=11\n"
         "ngram 2=8\n\n"
         "\\1-grams:\n"
         "-5\t<unk>\t0\n"
         "-1\t<s>\t0\n"
         "-1\t</s>\t0\n"
         "-2\tA\t0\n"
         "-2\tB\t0\n"
         "-2\tC\t0\n"
         "-2\tD\t0\n"
         "-2\tF\t0\n"
         "-2\tG\t0\n"
         "-2\tNEXT\t0\n"
         "-2\tOTHER\t0\n\n"
         "\\2-grams:\n"
         "-4\t<s> A\n"
         "-4\tA B\n"
         "-4\tB C\n"
         "-4\tC NEXT\n"
         "-0.1\t<s> D\n"
         "-0.1\tD F\n"
         "-0.1\tF G\n"
         "-0.1\tG NEXT\n\n"
         "\\end\\\n";
}

fl::lib::text::Dictionary MakeFlWords(const WordDictionary& words) {
  fl::lib::text::Dictionary result;
  for (int id = 0; id < words.Size(); ++id) {
    result.addEntry(words.Word(id), id);
  }
  result.setDefaultIndex(words.Id("<unk>"));
  return result;
}

std::pair<fl::lib::text::LMStatePtr, float> ScoreSequence(
    MaxFusionLm* lm, const std::vector<int>& words) {
  auto state = lm->start(false);
  float total = 0.0f;
  for (int word : words) {
    auto step = lm->score(state, word);
    state = std::move(step.first);
    total += step.second;
  }
  return {std::move(state), total};
}

}  // namespace

int main() {
  const fs::path dir = fs::temp_directory_path() /
      ("asr_sdk_pre_lm_mapping_test_" + std::to_string(getpid()));
  try {
    fs::remove_all(dir);
    fs::create_directories(dir);
    Write(dir / "words.txt",
          "<unk> 0\nA 1\nB 2\nC 3\nD 4\nF 5\nG 6\n"
          "NEXT 7\nOTHER 8\n");
    Write(dir / "lm.arpa", Arpa());
    Write(dir / "mapping.txt", "A B C -> D F G\n");
    Write(dir / "short_mapping.txt", "A B C -> D\n");

    WordDictionary words(dir / "words.txt");
    const auto fl_words = MakeFlWords(words);
    FixedLmConfig first;
    first.filename = "first.bin";
    first.path = dir / "lm.arpa";
    first.type = FixedLmType::kNgram;
    first.weight = 1.0;
    first.clip = false;
    FixedLmConfig second = first;
    second.filename = "second.bin";
    second.weight = 0.7;

    MaxFusionLm baseline({first, second}, fl_words, nullptr, -2.0f);
    auto target = ScoreSequence(&baseline, {4, 5, 6});
    auto target_next = baseline.score(target.first, 7);
    auto source = ScoreSequence(&baseline, {1, 2, 3});
    auto source_next = baseline.score(source.first, 7);

    auto mapper = std::make_shared<OutputSequenceMapper>(
        OutputSequenceMapper::Load(dir / "mapping.txt", words));
    mapper->ValidateForPreLm(dir / "mapping.txt");
    MaxFusionLm mapped({first, second}, fl_words, nullptr, -2.0f, mapper);
    auto replayed = ScoreSequence(&mapped, {1, 2, 3});
    Near(replayed.second, target.second,
         "source LM contributions were not replaced by mapped contributions");
    auto replayed_next = mapped.score(replayed.first, 7);
    Near(replayed_next.second, target_next.second,
         "future word did not use the mapped LM history");
    if (std::fabs(replayed_next.second - source_next.second) < 1e-3f) {
      throw std::runtime_error(
          "test LM does not distinguish mapped and raw future histories");
    }

    auto short_mapper = std::make_shared<OutputSequenceMapper>(
        OutputSequenceMapper::Load(dir / "short_mapping.txt", words));
    short_mapper->ValidateForPreLm(dir / "short_mapping.txt");
    MaxFusionLm shortened({first, second}, fl_words, nullptr, -2.0f,
                          short_mapper);
    const auto shortened_source = ScoreSequence(&shortened, {1, 2, 3});
    const auto one_target = ScoreSequence(&baseline, {4});
    // Three raw words receive Flashlight's external word score. Replaying one
    // logical target therefore returns +4 here to cancel two extra -2 scores.
    Near(shortened_source.second, one_target.second + 4.0f,
         "mapped/source length-penalty correction is wrong");

    fs::remove_all(dir);
    std::cout << "pre_lm_mapping_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(dir);
    std::cerr << "pre_lm_mapping_test failed: " << e.what() << "\n";
    return 1;
  }
}
