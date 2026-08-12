#include "flashlight_decoder/flashlight_result_mapper.h"

#include <algorithm>
#include <cctype>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "flashlight/lib/text/decoder/lm/KenLM.h"
#include "flashlight_decoder/contact_bias.h"
#include "flashlight_decoder/dynamic_contact_lexicon.h"
#include "flashlight_decoder/max_fusion_lm.h"

namespace asr_sdk::internal::flashlight_decoder {
namespace {

std::vector<std::string> SplitWhitespace(const std::string& value) {
  std::istringstream input(value);
  std::vector<std::string> words;
  std::string word;
  while (input >> word) {
    words.push_back(word);
  }
  return words;
}

void AppendLmTrace(const std::string& event,
                   const fl::lib::text::LMStatePtr& state,
                   DecodedHypothesis* hyp) {
  const auto fusion = std::static_pointer_cast<MaxFusionLmState>(state);
  if (!fusion || fusion->winner_filename.empty()) return;
  hyp->lm_events.push_back(LmEventTrace{
      event, fusion->winner_filename, fusion->winner_type,
      fusion->raw_score, fusion->reference_score, fusion->adjusted_score,
      fusion->weighted_score});
}

void AppendContactWords(const DecodedWord& source,
                        std::vector<DecodedWord>* output) {
  std::vector<std::string> visible_words = SplitWhitespace(source.text);
  if (visible_words.empty()) {
    visible_words.push_back(source.text);
  }
  const int width = std::max(0, source.end_frame - source.start_frame);
  for (size_t index = 0; index < visible_words.size(); ++index) {
    DecodedWord word;
    word.word_id = -1;
    word.text = visible_words[index];
    word.is_contact = true;
    word.contact_entity_index = source.contact_entity_index;
    word.start_frame = source.start_frame +
                       static_cast<int>((width * index) / visible_words.size());
    word.end_frame =
        source.start_frame +
        static_cast<int>((width * (index + 1)) / visible_words.size());
    word.timestamp_derived = true;
    output->push_back(std::move(word));
  }
}

std::vector<DecodedWord> MapContactAwareWords(
    const std::vector<DecodedWord>& raw,
    const FlashlightDecoderResource& resource) {
  std::vector<DecodedWord> output;
  size_t begin = 0;
  while (begin < raw.size()) {
    if (raw[begin].is_contact) {
      AppendContactWords(raw[begin], &output);
      ++begin;
      continue;
    }
    size_t end = begin;
    while (end < raw.size() && !raw[end].is_contact) {
      ++end;
    }
    std::vector<DecodedWord> span(raw.begin() + begin, raw.begin() + end);
    std::vector<DecodedWord> mapped = resource.Mapper().RewriteWords(span);
    output.insert(output.end(), std::make_move_iterator(mapped.begin()),
                  std::make_move_iterator(mapped.end()));
    begin = end;
  }
  return output;
}

}  // namespace

DecodedHypothesis ConvertFlashlightResult(
    const fl::lib::text::DecodeResult& result,
    const FlashlightDecoderResource& resource) {
  DecodedHypothesis hyp;
  hyp.first_pass_score = result.score;
  hyp.total_score = result.score;
  hyp.am_score = result.emittingModelScore;
  hyp.lm_score = result.lmScore;
  hyp.weighted_lm_score = result.lmScore;
  hyp.token_ids.reserve(result.tokens.size());
  for (int token : result.tokens) {
    if (token >= 0) {
      hyp.token_ids.push_back(token);
    }
  }

  // Keep the exact pre-contact conversion path for ordinary resources.  This
  // is intentional: no-context decoding must not route through the wrapper.
  if (!resource.HasSlotContext()) {
    auto replay_state = resource.WordLm()->start(false);
    for (int frame = 0; frame < static_cast<int>(result.words.size()); ++frame) {
      const int word_id = result.words[static_cast<size_t>(frame)];
      if (word_id < 0) {
        continue;
      }
      DecodedWord word;
      word.word_id = word_id;
      word.text = resource.OutputWords().Word(word_id);
      float ignored = 0.0f;
      std::tie(replay_state, ignored) =
          resource.WordLm()->score(replay_state, word_id);
      AppendLmTrace(word.text, replay_state, &hyp);
      word.start_frame = frame;
      word.end_frame = frame + 1;
      hyp.raw_words.push_back(std::move(word));
    }
    hyp.am_mapped_words = hyp.raw_words;
    hyp.mapped_words = resource.Mapper().RewriteWords(hyp.raw_words);
    return hyp;
  }

  const auto& contacts = resource.DynamicSlots();
  if (!contacts) {
    throw std::runtime_error("slot decoder resource has no dynamic registry");
  }
  auto replay_state = resource.WordLm()->start(false);
  for (int frame = 0; frame < static_cast<int>(result.words.size()); ++frame) {
    const int word_id = result.words[static_cast<size_t>(frame)];
    if (word_id < 0) {
      continue;
    }
    DecodedWord word;
    word.word_id = word_id;
    word.start_frame = frame;
    word.end_frame = frame + 1;
    float replay_contribution = 0.0f;
    std::tie(replay_state, replay_contribution) =
        resource.WordLm()->score(replay_state, word_id);
    if (contacts->IsDynamicContactId(word_id)) {
      const RuntimeContactForm& form = contacts->ContactFormForId(word_id);
      AppendLmTrace(form.slot_token, replay_state, &hyp);
      DecodedEntity entity;
      entity.slot_token = form.slot_token;
      entity.type = form.slot_token.size() > 2
                        ? form.slot_token.substr(1, form.slot_token.size() - 2)
                        : "slot";
      std::transform(entity.type.begin(), entity.type.end(), entity.type.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      entity.text = form.visible_text;
      entity.start_frame = word.start_frame;
      entity.end_frame = word.end_frame;
      entity.score = replay_contribution - ContactWordScoreCorrection(
                                               form.logical_word_count,
                                               resource.Options().word_score);
      entity.ambiguous = form.candidates.size() > 1;
      entity.candidates.reserve(form.candidates.size());
      for (const RuntimeContactCandidate& candidate : form.candidates) {
        entity.candidates.push_back(
            DecodedContactCandidate{candidate.value_id, candidate.contact_id,
                                    candidate.display_name});
      }
      word.text = entity.text;
      word.is_contact = true;
      word.contact_entity_index = static_cast<int>(hyp.entities.size());
      hyp.entities.push_back(std::move(entity));
    } else {
      if (!resource.OutputWords().ContainsId(word_id)) {
        throw std::runtime_error("unknown non-contact word id in decoder result: " +
                                 std::to_string(word_id));
      }
      word.text = resource.OutputWords().Word(word_id);
      AppendLmTrace(word.text, replay_state, &hyp);
    }
    hyp.raw_words.push_back(std::move(word));
  }
  hyp.am_mapped_words = hyp.raw_words;
  hyp.mapped_words = MapContactAwareWords(hyp.raw_words, resource);
  return hyp;
}

}  // namespace asr_sdk::internal::flashlight_decoder
