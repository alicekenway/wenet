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
#include "package/model_package.h"
#include "sherpa_onnx_wenet/token_table.h"

namespace asr_sdk::internal::flashlight_decoder {

class DynamicContactLexicon;
struct SharedFixedLmResources;

class FlashlightDecoderResource {
 public:
  FlashlightDecoderResource(
      const std::filesystem::path& tokens_path,
      const std::filesystem::path& words_path,
      const std::filesystem::path& lexicon_path,
      std::vector<FixedLmConfig> fixed_lms,
      const std::filesystem::path& pre_lm_mapping_path,
      const std::filesystem::path& final_mapping_path,
      FlashlightDecoderOptions options, std::string blank_token,
      std::string sil_token, std::string unk_word,
      double length_penalty = 0.0);

  // Creates a slot-enabled overlay resource. Base token/word tables, the
  // package mapper, and every fixed LM are retained; only the trie and active
  // composite LM are replaced. The base resource itself is never modified.
  static std::shared_ptr<const FlashlightDecoderResource>
  CreateSlotContextResource(
      const FlashlightDecoderResource& base,
      std::shared_ptr<fl::lib::text::Trie> combined_trie,
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
  const std::vector<FixedLmConfig>& FixedLms() const { return fixed_lms_; }
  const OutputSequenceMapper& PreLmMapper() const {
    return *pre_lm_mapper_;
  }
  const OutputSequenceMapper& FinalMapper() const {
    return final_output_mapper_;
  }
  const FlashlightDecoderOptions& Options() const { return options_; }

  int BlankId() const { return blank_id_; }
  int SilenceId() const { return sil_id_; }
  int UnknownWordId() const { return unk_word_id_; }
  int LexiconEntryCount() const { return lexicon_entry_count_; }
  int BaseWordCount() const { return output_words_.Size(); }
  bool SupportsRuntimeSlots() const;
  bool SupportsRuntimeSlot(const std::string& slot) const;
  const std::vector<LexiconEntry>& BaseLexiconEntries() const {
    return base_lexicon_entries_;
  }
  const std::unordered_map<std::string, std::vector<std::vector<int>>>&
  WordSpellings() const {
    return word_spellings_;
  }
  const std::shared_ptr<const DynamicContactLexicon>& DynamicSlots() const {
    return dynamic_contacts_;
  }
  bool HasSlotContext() const { return dynamic_contacts_ != nullptr; }

 private:
  FlashlightDecoderResource(
      const FlashlightDecoderResource& base,
      std::shared_ptr<fl::lib::text::Trie> combined_trie,
      std::shared_ptr<const DynamicContactLexicon> dynamic_contacts);

  sherpa_onnx_wenet::TokenTable am_tokens_;
  WordDictionary output_words_;
  fl::lib::text::Dictionary token_fl_dict_;
  fl::lib::text::Dictionary word_fl_dict_;
  std::shared_ptr<fl::lib::text::Trie> lexicon_trie_;
  fl::lib::text::LMPtr word_lm_;
  std::shared_ptr<const SharedFixedLmResources> fixed_lm_resources_;
  std::vector<FixedLmConfig> fixed_lms_;
  std::shared_ptr<const OutputSequenceMapper> pre_lm_mapper_;
  OutputSequenceMapper final_output_mapper_;
  FlashlightDecoderOptions options_;
  int blank_id_ = 0;
  int sil_id_ = 0;
  int unk_word_id_ = 0;
  int lexicon_entry_count_ = 0;
  std::vector<LexiconEntry> base_lexicon_entries_;
  std::unordered_map<std::string, std::vector<std::vector<int>>>
      word_spellings_;
  std::shared_ptr<const DynamicContactLexicon> dynamic_contacts_;
};

using FlashlightDecoderResourcePtr =
    std::shared_ptr<const FlashlightDecoderResource>;

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_DECODER_RESOURCE_H_
