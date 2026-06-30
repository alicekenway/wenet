#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_DECODER_RESOURCE_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_DECODER_RESOURCE_H_

#include <filesystem>
#include <memory>
#include <string>

#include "flashlight/lib/text/decoder/Trie.h"
#include "flashlight/lib/text/decoder/lm/LM.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "flashlight_decoder/flashlight_decoder_options.h"
#include "flashlight_decoder/output_sequence_mapper.h"
#include "flashlight_decoder/word_dictionary.h"
#include "sherpa_onnx_wenet/token_table.h"

namespace asr_sdk::internal::flashlight_decoder {

class FlashlightDecoderResource {
 public:
  FlashlightDecoderResource(
      const std::filesystem::path& tokens_path,
      const std::filesystem::path& words_path,
      const std::filesystem::path& lexicon_path,
      const std::filesystem::path& lm_path,
      const std::filesystem::path& am_mapping_path,
      const std::filesystem::path& final_mapping_path,
      FlashlightDecoderOptions options, std::string blank_token,
      std::string sil_token, std::string unk_word);

  const sherpa_onnx_wenet::TokenTable& AmTokens() const { return am_tokens_; }
  const WordDictionary& OutputWords() const { return output_words_; }
  const fl::lib::text::Dictionary& WordFlDictionary() const {
    return word_fl_dict_;
  }
  const std::shared_ptr<fl::lib::text::Trie>& LexiconTrie() const {
    return lexicon_trie_;
  }
  const fl::lib::text::LMPtr& WordLm() const { return word_lm_; }
  const OutputSequenceMapper& AmMapper() const { return am_mapper_; }
  const OutputSequenceMapper& FinalMapper() const { return final_mapper_; }
  const OutputSequenceMapper& Mapper() const { return final_mapper_; }
  const FlashlightDecoderOptions& Options() const { return options_; }

  int BlankId() const { return blank_id_; }
  int SilenceId() const { return sil_id_; }
  int UnknownWordId() const { return unk_word_id_; }
  int LexiconEntryCount() const { return lexicon_entry_count_; }

 private:
  sherpa_onnx_wenet::TokenTable am_tokens_;
  WordDictionary output_words_;
  fl::lib::text::Dictionary token_fl_dict_;
  fl::lib::text::Dictionary word_fl_dict_;
  std::shared_ptr<fl::lib::text::Trie> lexicon_trie_;
  fl::lib::text::LMPtr word_lm_;
  OutputSequenceMapper am_mapper_;
  OutputSequenceMapper final_mapper_;
  FlashlightDecoderOptions options_;
  int blank_id_ = 0;
  int sil_id_ = 0;
  int unk_word_id_ = 0;
  int lexicon_entry_count_ = 0;
};

using FlashlightDecoderResourcePtr =
    std::shared_ptr<const FlashlightDecoderResource>;

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_DECODER_RESOURCE_H_
