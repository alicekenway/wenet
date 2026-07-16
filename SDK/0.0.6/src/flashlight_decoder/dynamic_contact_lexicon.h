#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_DYNAMIC_CONTACT_LEXICON_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_DYNAMIC_CONTACT_LEXICON_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "asr_sdk/decode_context.h"
#include "asr_sdk/status.h"
#include "flashlight/lib/text/decoder/Trie.h"
#include "flashlight/lib/text/decoder/lm/LM.h"
#include "flashlight_decoder/contact_spelling_compiler.h"

namespace asr_sdk::internal::flashlight_decoder {

class FlashlightDecoderResource;

struct RuntimeContactCandidate {
  std::string contact_id;
  std::string display_name;
};

struct RuntimeContactForm {
  int dynamic_word_id = -1;
  std::string spoken_form;
  std::string visible_text;
  std::vector<int> token_ids;
  int logical_word_count = 1;
  std::vector<RuntimeContactCandidate> candidates;
};

// Immutable metadata for the dynamic labels inserted into one decode context.
class DynamicContactLexicon {
 public:
  static StatusOr<std::shared_ptr<const DynamicContactLexicon>> Create(
      const FlashlightDecoderResource& base,
      std::vector<CompiledContactSpelling> spellings);

  bool IsDynamicContactId(int word_id) const;
  const RuntimeContactForm& ContactFormForId(int word_id) const;
  std::string InternalWordForId(int word_id) const;
  int BaseWordCount() const { return base_word_count_; }
  const std::vector<RuntimeContactForm>& Forms() const { return forms_; }

  // Builds a fresh composite-LM trie so the base/main-LM trie remains
  // immutable.  Ordinary entries use main-LM lookahead; dynamic contact forms
  // use the largest configured pattern bonus as a safe pruning lookahead.
  std::shared_ptr<fl::lib::text::Trie> BuildCombinedTrie(
      const FlashlightDecoderResource& base) const;

 private:
  int base_word_count_ = 0;
  std::vector<RuntimeContactForm> forms_;
  std::unordered_map<int, size_t> form_index_by_id_;
};

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_DYNAMIC_CONTACT_LEXICON_H_
