#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

#include "flashlight/lib/text/decoder/LexiconDecoder.h"
#include "flashlight/lib/text/decoder/Trie.h"
#include "flashlight/lib/text/decoder/lm/ZeroLM.h"
#include "flashlight_decoder/compact_lexicon.h"
#include "flashlight_decoder/compact_lexicon_builder.h"
#include "flashlight_decoder/multi_trie_lexicon_decoder.h"

namespace {

using asr_sdk::internal::flashlight_decoder::MultiTrieLexiconDecoder;
using asr_sdk::internal::flashlight_decoder::RuntimeTrieLookahead;
using asr_sdk::internal::flashlight_decoder::CompactLexicon;
using asr_sdk::internal::flashlight_decoder::LexiconEntry;
using fl::lib::text::DecodeResult;

void Expect(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}

fl::lib::text::LexiconDecoderOptions Options() {
  fl::lib::text::LexiconDecoderOptions options;
  options.beamSize = 10000;
  options.beamSizeToken = 5;
  options.beamThreshold = 1000.0;
  options.lmWeight = 1.0;
  options.wordScore = 0.0;
  options.unkScore = -std::numeric_limits<float>::infinity();
  options.silScore = 0.0;
  options.logAdd = false;
  options.criterionType = fl::lib::text::CriterionType::CTC;
  return options;
}

std::vector<DecodeResult> Run(fl::lib::text::Decoder* decoder,
                              const std::vector<int>& best_tokens) {
  constexpr int kTokenCount = 5;
  std::vector<float> emissions(best_tokens.size() * kTokenCount, -20.0f);
  for (size_t frame = 0; frame < best_tokens.size(); ++frame) {
    emissions[frame * kTokenCount + best_tokens[frame]] = 0.0f;
  }
  decoder->decodeBegin();
  decoder->decodeStep(emissions.data(), best_tokens.size(), kTokenCount);
  decoder->decodeEnd();
  return decoder->getAllFinalHypothesis();
}

std::vector<std::pair<std::vector<int>, double>> Canonical(
    const std::vector<DecodeResult>& results) {
  std::vector<std::pair<std::vector<int>, double>> out;
  for (const auto& result : results) {
    std::vector<int> words;
    for (int word : result.words) {
      if (word >= 0) words.push_back(word);
    }
    out.emplace_back(std::move(words), result.score);
  }
  std::sort(out.begin(), out.end(), [](const auto& left, const auto& right) {
    if (left.first != right.first) return left.first < right.first;
    return left.second < right.second;
  });
  return out;
}

void Compare(const std::vector<DecodeResult>& combined_results,
             const std::vector<DecodeResult>& multi_results,
             const std::string& scenario) {
  const auto combined = Canonical(combined_results);
  const auto multi = Canonical(multi_results);
  if (combined.size() != multi.size()) {
    auto describe = [](const auto& values) {
      std::string text;
      for (const auto& value : values) {
        text += " [";
        for (int word : value.first) text += std::to_string(word) + ",";
        text += "]@" + std::to_string(value.second);
      }
      return text;
    };
    throw std::runtime_error(
        scenario + ": beam size differs combined=" +
        std::to_string(combined.size()) + describe(combined) + " multi=" +
        std::to_string(multi.size()) + describe(multi));
  }
  for (size_t i = 0; i < combined.size(); ++i) {
    Expect(combined[i].first == multi[i].first,
           scenario + ": emitted word sequence differs");
    Expect(std::fabs(combined[i].second - multi[i].second) < 1e-5,
           scenario + ": final score differs");
  }
}

}  // namespace

int main() {
  try {
    constexpr int kBlank = 0;
    constexpr int kSil = 1;
    auto base = std::make_shared<fl::lib::text::Trie>(5, kSil);
    // Base trie keeps a legacy score and a distinct runtime strict-max score.
    base->insert({2, 3}, 10, 7.0f);
    base->insert({2, 4}, 11, 6.0f);
    base->smear(fl::lib::text::SmearingMode::MAX);
    auto lookahead = std::make_shared<RuntimeTrieLookahead>();
    (*lookahead)[base->search({2}).get()] = -2.0f;
    (*lookahead)[base->search({2, 3}).get()] = -2.0f;
    (*lookahead)[base->search({2, 4}).get()] = -3.0f;
    const std::filesystem::path compact_path =
        std::filesystem::temp_directory_path() /
        ("compact_decoder_test_" + std::to_string(getpid()) + ".bin");
    std::vector<LexiconEntry> base_entries;
    LexiconEntry first;
    first.word_id = 10;
    first.word = "first";
    first.token_ids = {2, 3};
    base_entries.push_back(first);
    LexiconEntry second;
    second.word_id = 11;
    second.word = "second";
    second.token_ids = {2, 4};
    base_entries.push_back(second);
    asr_sdk::internal::flashlight_decoder::WriteCompactLexicon(
        compact_path, base, *lookahead, base_entries, 5, 30, kSil, kBlank,
        0, 0);
    const auto compact = CompactLexicon::Load(compact_path);

    auto empty_overlay = std::make_shared<fl::lib::text::Trie>(5, kSil);
    fl::lib::text::LexiconDecoder old_base_decoder(
        Options(), base, std::make_shared<fl::lib::text::ZeroLM>(),
        kSil, kBlank, 0, {}, false);
    MultiTrieLexiconDecoder compact_base_decoder(
        Options(), compact, empty_overlay,
        std::make_shared<fl::lib::text::ZeroLM>(), kSil, kBlank, 0, {},
        false, false);
    Compare(Run(&old_base_decoder, {2, 0, 3}),
            Run(&compact_base_decoder, {2, 0, 3}), "base-only");

    auto overlay = std::make_shared<fl::lib::text::Trie>(5, kSil);
    // One contact is a base-word homophone; another is both a complete name
    // and a prefix of a longer contact.
    overlay->insert({2, 3}, 20, 0.0f);
    overlay->insert({2}, 21, 0.0f);
    overlay->insert({2, 4, 3}, 22, 0.0f);
    overlay->smear(fl::lib::text::SmearingMode::RUNTIME_CONTACT_MAX);

    auto combined = std::make_shared<fl::lib::text::Trie>(5, kSil);
    combined->insert({2, 3}, 10, -2.0f);
    combined->insert({2, 4}, 11, -3.0f);
    combined->insert({2, 3}, 20, 0.0f);
    combined->insert({2}, 21, 0.0f);
    combined->insert({2, 4, 3}, 22, 0.0f);
    combined->smear(fl::lib::text::SmearingMode::RUNTIME_CONTACT_MAX);

    auto combined_lm = std::make_shared<fl::lib::text::ZeroLM>();
    fl::lib::text::LexiconDecoder old_decoder(
        Options(), combined, combined_lm, kSil, kBlank, -1, {}, false);
    auto multi_lm = std::make_shared<fl::lib::text::ZeroLM>();
    MultiTrieLexiconDecoder new_decoder(
        Options(), compact, overlay, multi_lm,
        kSil, kBlank, 0, {}, false, true);
    Compare(Run(&old_decoder, {2, 0, 3}),
            Run(&new_decoder, {2, 0, 3}), "homophone");

    fl::lib::text::LexiconDecoder old_prefix_decoder(
        Options(), combined, std::make_shared<fl::lib::text::ZeroLM>(),
        kSil, kBlank, -1, {}, false);
    MultiTrieLexiconDecoder new_prefix_decoder(
        Options(), compact, overlay,
        std::make_shared<fl::lib::text::ZeroLM>(), kSil, kBlank, 0, {},
        false, true);
    Compare(Run(&old_prefix_decoder, {2, 0, 4, 0, 3}),
            Run(&new_prefix_decoder, {2, 0, 4, 0, 3}),
            "contact-prefix");
    std::filesystem::remove(compact_path);
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
