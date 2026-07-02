#include "flashlight_decoder/kenlm_rescorer.h"

#include <tuple>
#include <utility>

namespace asr_sdk::internal::flashlight_decoder {
namespace {

int CountWords(const std::vector<DecodedWord>& words) {
  int count = 0;
  for (const DecodedWord& word : words) {
    if (word.word_id >= 0) {
      ++count;
    }
  }
  return count;
}

int CountWordId(const std::vector<DecodedWord>& words, int word_id) {
  int count = 0;
  for (const DecodedWord& word : words) {
    if (word.word_id == word_id) {
      ++count;
    }
  }
  return count;
}

}  // namespace

double ScoreWordsWithKenLm(const FlashlightDecoderResource& resource,
                           const std::vector<DecodedWord>& words) {
  const auto& lm = resource.WordLm();
  auto state = lm->start(false);
  double score = 0.0;

  for (const DecodedWord& word : words) {
    if (word.word_id < 0) {
      continue;
    }
    fl::lib::text::LMStatePtr next_state;
    float word_score = 0.0f;
    std::tie(next_state, word_score) = lm->score(state, word.word_id);
    state = std::move(next_state);
    score += static_cast<double>(word_score);
  }
  return score;
}

void RescoreAndApplyFinalMapping(const FlashlightDecoderResource& resource,
                                 std::vector<DecodedHypothesis>* hyps) {
  if (hyps == nullptr) {
    return;
  }
  const FlashlightDecoderOptions& options = resource.Options();
  for (DecodedHypothesis& hyp : *hyps) {
    hyp.lm_score = ScoreWordsWithKenLm(resource, hyp.am_mapped_words);
    const int word_count = CountWords(hyp.am_mapped_words);
    const int unk_count =
        CountWordId(hyp.am_mapped_words, resource.UnknownWordId());
    hyp.total_score =
        hyp.am_score + options.lm_weight * hyp.lm_score +
        options.word_score * static_cast<double>(word_count) +
        options.unk_score * static_cast<double>(unk_count);
    hyp.mapped_words = resource.FinalMapper().RewriteWords(hyp.am_mapped_words);
  }
}

}  // namespace asr_sdk::internal::flashlight_decoder
