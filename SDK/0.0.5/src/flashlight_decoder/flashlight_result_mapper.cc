#include "flashlight_decoder/flashlight_result_mapper.h"

namespace asr_sdk::internal::flashlight_decoder {

DecodedHypothesis ConvertFlashlightResult(
    const fl::lib::text::DecodeResult& result,
    const FlashlightDecoderResource& resource) {
  DecodedHypothesis hyp;
  hyp.first_pass_score = result.score;
  hyp.total_score = result.score;
  hyp.am_score = result.emittingModelScore;
  hyp.lm_score = result.lmScore;
  hyp.token_ids.reserve(result.tokens.size());
  for (int token : result.tokens) {
    if (token >= 0) {
      hyp.token_ids.push_back(token);
    }
  }

  for (int frame = 0; frame < static_cast<int>(result.words.size()); ++frame) {
    const int word_id = result.words[static_cast<size_t>(frame)];
    if (word_id < 0) {
      continue;
    }
    DecodedWord word;
    word.word_id = word_id;
    word.text = resource.OutputWords().Word(word_id);
    word.start_frame = frame;
    word.end_frame = frame + 1;
    hyp.raw_words.push_back(std::move(word));
  }
  hyp.am_mapped_words = hyp.raw_words;
  hyp.mapped_words = resource.Mapper().RewriteWords(hyp.raw_words);
  return hyp;
}

}  // namespace asr_sdk::internal::flashlight_decoder
