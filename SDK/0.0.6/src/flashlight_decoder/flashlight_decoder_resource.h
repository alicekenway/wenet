#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_DECODER_RESOURCE_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_DECODER_RESOURCE_H_

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "flashlight/lib/text/decoder/Trie.h"
#include "flashlight/lib/text/decoder/lm/LM.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight_decoder/flashlight_decoder_options.h"
#include "flashlight_decoder/lexicon_loader.h"
#include "flashlight_decoder/output_sequence_mapper.h"
#include "flashlight_decoder/word_dictionary.h"
#include "sherpa_onnx_wenet/token_table.h"

namespace asr_sdk::internal::flashlight_decoder {

class DynamicContactLexicon;

class FlashlightDecoderResource {
 public:
  FlashlightDecoderResource(
      const std::filesystem::path& tokens_path,
      const std::filesystem::path& words_path,
      const std::filesystem::path& lexicon_path,
      const std::filesystem::path& lm_path,
      const std::filesystem::path& mapping_path,
      FlashlightDecoderOptions options, std::string blank_token,
      std::string sil_token, std::string unk_word,
      std::string contact_class_word = "",
      std::filesystem::path contact_lm_path = {},
      double contact_lm_weight = 1.5,
      double contact_lm_accumulation_factor = 0.5,
      double contact_lm_max_bonus = 0.0);

  // Creates a contact-enabled overlay resource.  Base token/word tables, the
  // package mapper, and both package KenLMs are retained; only the trie and
  // active decoder LM are replaced.  The base resource itself is never
  // modified.
  static std::shared_ptr<const FlashlightDecoderResource>
  CreateContactContextResource(
      const FlashlightDecoderResource& base,
      std::shared_ptr<fl::lib::text::Trie> combined_trie,
      fl::lib::text::LMPtr contextual_lm,
      std::shared_ptr<const DynamicContactLexicon> dynamic_contacts);

  const sherpa_onnx_wenet::TokenTable& AmTokens() const { return am_tokens_; }
  const WordDictionary& OutputWords() const { return output_words_; }
  const fl::lib::text::Dictionary& WordFlDictionary() const {
    return word_fl_dict_;
  }
  const std::shared_ptr<fl::lib::text::Trie>& LexiconTrie() const {
    return lexicon_trie_;
  }
  const fl::lib::text::LMPtr& WordLm() const { return word_lm_; }
  // The unwrapped contact-domain KenLM.  It is null for main-only packages.
  const fl::lib::text::LMPtr& ContactWordLm() const {
    return contact_word_lm_;
  }
  double ContactLmWeight() const { return contact_lm_weight_; }
  double ContactLmAccumulationFactor() const {
    return contact_lm_accumulation_factor_;
  }
  double ContactLmMaxBonus() const { return contact_lm_max_bonus_; }
  const OutputSequenceMapper& Mapper() const { return output_mapper_; }
  const FlashlightDecoderOptions& Options() const { return options_; }

  int BlankId() const { return blank_id_; }
  int SilenceId() const { return sil_id_; }
  int UnknownWordId() const { return unk_word_id_; }
  int LexiconEntryCount() const { return lexicon_entry_count_; }
  int BaseWordCount() const { return output_words_.Size(); }
  bool SupportsRuntimeContacts() const {
    return contact_class_word_id_ >= 0 && contact_word_lm_ != nullptr;
  }
  int ContactClassWordId() const { return contact_class_word_id_; }
  const std::string& ContactClassWord() const { return contact_class_word_; }
  const std::vector<LexiconEntry>& BaseLexiconEntries() const {
    return base_lexicon_entries_;
  }
  const std::unordered_map<std::string, std::vector<std::vector<int>>>&
  WordSpellings() const {
    return word_spellings_;
  }
  const std::shared_ptr<const DynamicContactLexicon>& DynamicContacts() const {
    return dynamic_contacts_;
  }
  bool IsContactContext() const { return dynamic_contacts_ != nullptr; }

 private:
  FlashlightDecoderResource(
      const FlashlightDecoderResource& base,
      std::shared_ptr<fl::lib::text::Trie> combined_trie,
      fl::lib::text::LMPtr contextual_lm,
      std::shared_ptr<const DynamicContactLexicon> dynamic_contacts);

  sherpa_onnx_wenet::TokenTable am_tokens_;
  WordDictionary output_words_;
  fl::lib::text::Dictionary token_fl_dict_;
  fl::lib::text::Dictionary word_fl_dict_;
  std::shared_ptr<fl::lib::text::Trie> lexicon_trie_;
  fl::lib::text::LMPtr word_lm_;
  fl::lib::text::LMPtr contact_word_lm_;
  double contact_lm_weight_ = 1.5;
  double contact_lm_accumulation_factor_ = 0.5;
  double contact_lm_max_bonus_ = 0.0;
  OutputSequenceMapper output_mapper_;
  FlashlightDecoderOptions options_;
  int blank_id_ = 0;
  int sil_id_ = 0;
  int unk_word_id_ = 0;
  int lexicon_entry_count_ = 0;
  std::string contact_class_word_;
  int contact_class_word_id_ = -1;
  std::vector<LexiconEntry> base_lexicon_entries_;
  std::unordered_map<std::string, std::vector<std::vector<int>>>
      word_spellings_;
  std::shared_ptr<const DynamicContactLexicon> dynamic_contacts_;
};

using FlashlightDecoderResourcePtr =
    std::shared_ptr<const FlashlightDecoderResource>;

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_DECODER_RESOURCE_H_
