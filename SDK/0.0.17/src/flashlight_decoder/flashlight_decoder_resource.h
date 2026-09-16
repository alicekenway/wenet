#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_DECODER_RESOURCE_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_DECODER_RESOURCE_H_

#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "flashlight/lib/text/decoder/Trie.h"
#include "flashlight/lib/text/decoder/lm/LM.h"
#include "flashlight_decoder/compact_lexicon.h"
#include "flashlight_decoder/flashlight_decoder_options.h"
#include "flashlight_decoder/lexicon_loader.h"
#include "flashlight_decoder/output_sequence_mapper.h"
#include "flashlight_decoder/word_dictionary.h"
#include "package/model_package.h"
#include "sherpa_onnx_wenet/token_table.h"

namespace asr_sdk::internal::flashlight_decoder {

class DynamicContactLexicon;
class WordSpellingGenerator;
struct SharedDecoderResources;
struct SharedFixedLmResources;

using RuntimeTrieLookahead =
    std::unordered_map<const fl::lib::text::TrieNode*, float>;

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
      double length_penalty = 0.0, bool compact_binary = false,
      uint64_t expected_dependency_hash = 0,
      const std::filesystem::path& sentencepiece_model = {});

  // Creates a slot-enabled resource. All immutable package data, fixed LMs,
  // and the compact base lexicon are shared. Only the small overlay trie,
  // dynamic contact
  // metadata, and active composite-LM wrapper are context-specific.
  static std::shared_ptr<const FlashlightDecoderResource>
  CreateSlotContextResource(
      const FlashlightDecoderResource& base,
      std::shared_ptr<fl::lib::text::Trie> overlay_trie,
      std::shared_ptr<const DynamicContactLexicon> dynamic_contacts);

  const sherpa_onnx_wenet::TokenTable& AmTokens() const;
  const WordDictionary& OutputWords() const;
  const std::shared_ptr<const CompactLexicon>& Lexicon() const;
  // Legacy accessors exist only for the host-side compiler and parity tests.
  const std::shared_ptr<fl::lib::text::Trie>& LexiconTrie() const;
  const std::shared_ptr<fl::lib::text::Trie>& OverlayTrie() const {
    return overlay_trie_;
  }
  const fl::lib::text::LMPtr& WordLm() const { return word_lm_; }
  const std::vector<FixedLmConfig>& FixedLms() const;
  const OutputSequenceMapper& PreLmMapper() const;
  const OutputSequenceMapper& FinalMapper() const;
  const FlashlightDecoderOptions& Options() const;

  int BlankId() const;
  int SilenceId() const;
  int UnknownWordId() const;
  int LexiconEntryCount() const;
  int BaseWordCount() const;
  bool SupportsRuntimeSlots() const;
  bool SupportsRuntimeSlot(const std::string& slot) const;
  const std::vector<LexiconEntry>& BaseLexiconEntries() const;
  std::vector<std::vector<int>> WordSpellings(const std::string& word) const;
  // Lazy, thread-safe, shared by all contexts; never touches the base trie.
  const WordSpellingGenerator& ContactWordSpeller() const;
  std::shared_ptr<const RuntimeTrieLookahead>
  LegacyContextLookaheadForCompiler() const;
  const std::shared_ptr<const DynamicContactLexicon>& DynamicSlots() const {
    return dynamic_contacts_;
  }
  bool HasSlotContext() const { return dynamic_contacts_ != nullptr; }

 private:
  FlashlightDecoderResource(
      const FlashlightDecoderResource& base,
      std::shared_ptr<fl::lib::text::Trie> overlay_trie,
      std::shared_ptr<const DynamicContactLexicon> dynamic_contacts);

  std::shared_ptr<const SharedDecoderResources> shared_;
  std::shared_ptr<fl::lib::text::Trie> overlay_trie_;
  fl::lib::text::LMPtr word_lm_;
  std::shared_ptr<const DynamicContactLexicon> dynamic_contacts_;
};

using FlashlightDecoderResourcePtr =
    std::shared_ptr<const FlashlightDecoderResource>;

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_FLASHLIGHT_DECODER_RESOURCE_H_
