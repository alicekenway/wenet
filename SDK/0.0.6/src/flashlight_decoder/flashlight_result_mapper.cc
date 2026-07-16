#include "flashlight_decoder/flashlight_result_mapper.h"

#include <algorithm>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <tuple>

#include "flashlight/lib/text/decoder/lm/KenLM.h"
#include "flashlight_decoder/contact_bias.h"
#include "flashlight_decoder/dynamic_contact_lexicon.h"

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

class ContactLmReplay {
 public:
  explicit ContactLmReplay(const FlashlightDecoderResource& resource)
      : resource_(resource), contact_lm_(resource.ContactWordLm()) {
    if (!contact_lm_) {
      throw std::runtime_error("contact result mapping has no contact LM");
    }
    contact_kenlm_ = std::dynamic_pointer_cast<fl::lib::text::KenLM>(contact_lm_);
    if (!contact_kenlm_) {
      throw std::runtime_error("contact result mapping requires a KenLM");
    }
    state_ = contact_lm_->start(false);
  }

  float AdvanceContact(const RuntimeContactForm& form) {
    if (emitted_contact_) {
      throw std::runtime_error("multiple runtime contacts are not supported");
    }
    const auto matched = contact_kenlm_->scoreWithMetadata(
        state_, resource_.ContactClassWordId());
    if (matched.score >= 0.0f || matched.ngram_length < 2) {
      throw std::runtime_error("contact result has no matched pattern rule");
    }
    state_ = matched.state;
    emitted_contact_ = true;
    exact_standalone_contact_ =
        !saw_base_word_before_contact_ && matched.ngram_length == 2;
    return static_cast<float>(resource_.ContactLmWeight() * -matched.score) *
           ContactBonusMultiplier(form.logical_word_count,
                                  resource_.ContactLmAccumulationFactor());
  }

  void AdvanceBaseWord(int word_id) {
    if (exact_standalone_contact_) {
      throw std::runtime_error(
          "exact standalone contact was followed by another word");
    }
    if (contact_kenlm_->HasUserToken(word_id)) {
      float ignored_score = 0.0f;
      std::tie(state_, ignored_score) = contact_lm_->score(state_, word_id);
    } else {
      state_ = contact_lm_->start(true);
    }
    if (!emitted_contact_) {
      saw_base_word_before_contact_ = true;
    }
  }

 private:
  const FlashlightDecoderResource& resource_;
  fl::lib::text::LMPtr contact_lm_;
  std::shared_ptr<fl::lib::text::KenLM> contact_kenlm_;
  fl::lib::text::LMStatePtr state_;
  bool saw_base_word_before_contact_ = false;
  bool emitted_contact_ = false;
  bool exact_standalone_contact_ = false;
};

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
  hyp.source = resource.IsContactContext() ? DecoderSource::kContact
                                           : DecoderSource::kMain;
  hyp.weighted_lm_score = resource.IsContactContext()
                              ? result.lmScore
                              : result.lmScore * resource.Options().lm_weight;
  hyp.token_ids.reserve(result.tokens.size());
  for (int token : result.tokens) {
    if (token >= 0) {
      hyp.token_ids.push_back(token);
    }
  }

  // Keep the exact pre-contact conversion path for ordinary resources.  This
  // is intentional: no-context decoding must not route through the wrapper.
  if (!resource.IsContactContext()) {
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

  const auto& contacts = resource.DynamicContacts();
  if (!contacts) {
    throw std::runtime_error("contact decoder resource has no dynamic registry");
  }
  ContactLmReplay contact_lm_replay(resource);
  for (int frame = 0; frame < static_cast<int>(result.words.size()); ++frame) {
    const int word_id = result.words[static_cast<size_t>(frame)];
    if (word_id < 0) {
      continue;
    }
    DecodedWord word;
    word.word_id = word_id;
    word.start_frame = frame;
    word.end_frame = frame + 1;
    if (contacts->IsDynamicContactId(word_id)) {
      const RuntimeContactForm& form = contacts->ContactFormForId(word_id);
      DecodedEntity entity;
      entity.text = form.visible_text;
      entity.start_frame = word.start_frame;
      entity.end_frame = word.end_frame;
      entity.score = contact_lm_replay.AdvanceContact(form);
      entity.ambiguous = form.candidates.size() > 1;
      entity.candidates.reserve(form.candidates.size());
      for (const RuntimeContactCandidate& candidate : form.candidates) {
        entity.candidates.push_back(
            DecodedContactCandidate{candidate.contact_id, candidate.display_name});
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
      contact_lm_replay.AdvanceBaseWord(word_id);
    }
    hyp.raw_words.push_back(std::move(word));
  }
  hyp.am_mapped_words = hyp.raw_words;
  hyp.mapped_words = MapContactAwareWords(hyp.raw_words, resource);
  return hyp;
}

}  // namespace asr_sdk::internal::flashlight_decoder
