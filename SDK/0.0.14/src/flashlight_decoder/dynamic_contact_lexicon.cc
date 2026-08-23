#include "flashlight_decoder/dynamic_contact_lexicon.h"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "flashlight/lib/text/decoder/lm/KenLM.h"
#include "flashlight_decoder/flashlight_decoder_resource.h"

namespace asr_sdk::internal::flashlight_decoder {

StatusOr<std::shared_ptr<const DynamicContactLexicon>>
DynamicContactLexicon::Create(const FlashlightDecoderResource& base,
                              std::vector<CompiledContactSpelling> spellings) {
  auto out = std::make_shared<DynamicContactLexicon>();
  out->base_word_count_ = base.BaseWordCount();

  std::sort(spellings.begin(), spellings.end(), [](const auto& left,
                                                     const auto& right) {
    if (left.token_ids != right.token_ids) {
      return left.token_ids < right.token_ids;
    }
    if (left.slot_token != right.slot_token) {
      return left.slot_token < right.slot_token;
    }
    if (left.contact_id != right.contact_id) {
      return left.contact_id < right.contact_id;
    }
    if (left.spoken_form != right.spoken_form) {
      return left.spoken_form < right.spoken_form;
    }
    return left.display_name < right.display_name;
  });

  size_t begin = 0;
  while (begin < spellings.size()) {
    size_t end = begin + 1;
    while (end < spellings.size() &&
           spellings[end].token_ids == spellings[begin].token_ids &&
           spellings[end].slot_token == spellings[begin].slot_token) {
      ++end;
    }

    RuntimeContactForm form;
    form.dynamic_word_id = out->base_word_count_ +
                           static_cast<int>(out->forms_.size());
    form.slot_token = spellings[begin].slot_token;
    form.slot_word_id = spellings[begin].slot_word_id;
    form.spoken_form = spellings[begin].spoken_form;
    form.token_ids = spellings[begin].token_ids;
    form.logical_word_count = spellings[begin].logical_word_count;

    std::map<std::string, std::string> candidates_by_id;
    for (size_t index = begin; index < end; ++index) {
      if (spellings[index].logical_word_count != form.logical_word_count) {
        return Status::InvalidArgument(
            "identical AM contact spelling has conflicting "
            "logical_word_count values");
      }
      const auto [it, inserted] = candidates_by_id.emplace(
          spellings[index].contact_id, spellings[index].display_name);
      if (!inserted && it->second != spellings[index].display_name) {
        return Status::InvalidArgument(
            "same contact_id has inconsistent display names: " +
            spellings[index].contact_id);
      }
    }
    form.candidates.reserve(candidates_by_id.size());
    for (const auto& [contact_id, display_name] : candidates_by_id) {
      form.candidates.push_back(RuntimeContactCandidate{
          contact_id, form.slot_token == "<CONTACT>" ? contact_id : "",
          display_name});
    }
    form.visible_text = form.candidates.size() == 1
                            ? form.candidates.front().display_name
                            : form.spoken_form;
    out->form_index_by_id_[form.dynamic_word_id] = out->forms_.size();
    out->forms_.push_back(std::move(form));
    begin = end;
  }
  return std::shared_ptr<const DynamicContactLexicon>(std::move(out));
}

bool DynamicContactLexicon::IsDynamicContactId(int word_id) const {
  return form_index_by_id_.find(word_id) != form_index_by_id_.end();
}

const RuntimeContactForm& DynamicContactLexicon::ContactFormForId(
    int word_id) const {
  const auto it = form_index_by_id_.find(word_id);
  if (it == form_index_by_id_.end()) {
    throw std::out_of_range("unknown dynamic contact word id: " +
                            std::to_string(word_id));
  }
  return forms_[it->second];
}

std::string DynamicContactLexicon::InternalWordForId(int word_id) const {
  const RuntimeContactForm& form = ContactFormForId(word_id);
  return "__runtime_contact_form_" +
         std::to_string(form.dynamic_word_id - base_word_count_) + "__";
}

std::shared_ptr<fl::lib::text::Trie>
DynamicContactLexicon::BuildOverlayTrie(
    const FlashlightDecoderResource& base) const {
  auto trie = std::make_shared<fl::lib::text::Trie>(base.AmTokens().Size(),
                                                     base.SilenceId());
  for (const RuntimeContactForm& form : forms_) {
    // Contact bias must be added exactly once, at the verified virtual
    // <CONTACT> LM transition.  Supplying the maximum pattern bonus here as
    // trie lookahead can leak it into a completed form when that form shares
    // a terminal/prefix node with another contact.  A neutral score preserves
    // ordinary lexicon search and keeps contact-list size/prefix sharing from
    // changing the final LM score.
    trie->insert(form.token_ids, form.dynamic_word_id, 0.0f);
  }
  // Strict MAX keeps the overlay's shared prefix score independent of how
  // many contacts share it. The decoder takes max(base, overlay), which is
  // exactly the score produced by the former combined trie.
  trie->smear(fl::lib::text::SmearingMode::RUNTIME_CONTACT_MAX);
  return trie;
}

}  // namespace asr_sdk::internal::flashlight_decoder
