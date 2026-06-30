#include "flashlight_decoder/flashlight_decoder_resource.h"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <utility>

#include "flashlight/lib/text/decoder/lm/KenLM.h"
#include "flashlight_decoder/lexicon_loader.h"

namespace asr_sdk::internal::flashlight_decoder {
namespace {

fl::lib::text::Dictionary MakeTokenDictionary(
    const sherpa_onnx_wenet::TokenTable& tokens) {
  fl::lib::text::Dictionary dict;
  for (int id = 0; id < tokens.Size(); ++id) {
    dict.addEntry(tokens.Token(id), id);
  }
  return dict;
}

fl::lib::text::Dictionary MakeWordDictionary(const WordDictionary& words) {
  fl::lib::text::Dictionary dict;
  for (int id = 0; id < words.Size(); ++id) {
    dict.addEntry(words.Word(id), id);
  }
  dict.setDefaultIndex(words.Id("<unk>"));
  return dict;
}

fl::lib::text::SmearingMode SmearingModeFromString(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  if (value == "none") {
    return fl::lib::text::SmearingMode::NONE;
  }
  if (value == "max") {
    return fl::lib::text::SmearingMode::MAX;
  }
  if (value == "logadd" || value == "log_add") {
    return fl::lib::text::SmearingMode::LOGADD;
  }
  throw std::runtime_error("unknown Flashlight trie smearing mode: " + value);
}

}  // namespace

FlashlightDecoderResource::FlashlightDecoderResource(
    const std::filesystem::path& tokens_path,
    const std::filesystem::path& words_path,
    const std::filesystem::path& lexicon_path,
    const std::filesystem::path& lm_path,
    const std::filesystem::path& am_mapping_path,
    const std::filesystem::path& final_mapping_path,
    FlashlightDecoderOptions options,
    std::string blank_token, std::string sil_token, std::string unk_word)
    : am_tokens_(tokens_path),
      output_words_(words_path),
      token_fl_dict_(MakeTokenDictionary(am_tokens_)),
      word_fl_dict_(MakeWordDictionary(output_words_)),
      options_(std::move(options)) {
  if (!am_tokens_.Contains(blank_token)) {
    throw std::runtime_error("blank token not found in tokens.txt: " +
                             blank_token);
  }
  if (!am_tokens_.Contains(sil_token)) {
    throw std::runtime_error("silence/separator token not found in tokens.txt: " +
                             sil_token);
  }
  if (!output_words_.Contains(unk_word)) {
    throw std::runtime_error("unknown word not found in words.txt: " +
                             unk_word);
  }
  blank_id_ = am_tokens_.Id(blank_token);
  sil_id_ = am_tokens_.Id(sil_token);
  unk_word_id_ = output_words_.Id(unk_word);

  word_lm_ = std::make_shared<fl::lib::text::KenLM>(lm_path.string(),
                                                     word_fl_dict_);
  lexicon_trie_ =
      std::make_shared<fl::lib::text::Trie>(am_tokens_.Size(), sil_id_);

  const auto entries = LoadLexicon(lexicon_path, output_words_, am_tokens_);
  for (const LexiconEntry& entry : entries) {
    lexicon_trie_->insert(entry.token_ids, entry.word_id, 0.0f);
  }
  lexicon_entry_count_ = static_cast<int>(entries.size());
  lexicon_trie_->smear(SmearingModeFromString(options_.smearing));
  am_mapper_ = OutputSequenceMapper::Load(am_mapping_path, output_words_);
  final_mapper_ = OutputSequenceMapper::Load(final_mapping_path, output_words_);
}

}  // namespace asr_sdk::internal::flashlight_decoder
