#include "flashlight_decoder/flashlight_decoder_resource.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "flashlight/lib/text/decoder/lm/KenLM.h"
#include "flashlight_decoder/contact_bias.h"
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
    const std::filesystem::path& mapping_path, FlashlightDecoderOptions options,
    std::string blank_token, std::string sil_token, std::string unk_word,
    std::string contact_class_word, std::filesystem::path contact_lm_path,
    double contact_lm_weight, double contact_lm_accumulation_factor,
    double contact_lm_max_bonus)
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
  if (!contact_class_word.empty()) {
    if (!output_words_.Contains(contact_class_word)) {
      throw std::runtime_error("contact class word not found in words.txt: " +
                               contact_class_word);
    }
    if (contact_class_word == unk_word) {
      throw std::runtime_error(
          "contact class word must differ from the unknown word");
    }
    if (am_tokens_.Contains(contact_class_word)) {
      throw std::runtime_error(
          "contact class word must not be an AM token: " +
          contact_class_word);
    }
    contact_class_word_ = std::move(contact_class_word);
    contact_class_word_id_ = output_words_.Id(contact_class_word_);
  }

  word_lm_ = std::make_shared<fl::lib::text::KenLM>(lm_path.string(),
                                                     word_fl_dict_);
  if (!contact_lm_path.empty()) {
    if (contact_class_word_id_ < 0) {
      throw std::runtime_error(
          "contact LM requires a declared contact class word");
    }
    if (!std::isfinite(contact_lm_weight) || contact_lm_weight <= 0.0) {
      throw std::runtime_error(
          "contact LM weight must be finite and greater than zero");
    }
    if (!IsValidContactAccumulationFactor(contact_lm_accumulation_factor)) {
      throw std::runtime_error(
          "contact LM accumulation factor must be between zero and one");
    }
    if (!std::isfinite(contact_lm_max_bonus) || contact_lm_max_bonus <= 0.0) {
      throw std::runtime_error(
          "contact LM max bonus must be finite and greater than zero");
    }
    contact_word_lm_ = std::make_shared<fl::lib::text::KenLM>(
        contact_lm_path.string(), word_fl_dict_);
    if (!std::static_pointer_cast<fl::lib::text::KenLM>(contact_word_lm_)
             ->HasWord(contact_class_word_)) {
      throw std::runtime_error(
          "contact class word is missing from the contact KenLM vocabulary: " +
          contact_class_word_);
    }
    contact_lm_weight_ = contact_lm_weight;
    contact_lm_accumulation_factor_ = contact_lm_accumulation_factor;
    contact_lm_max_bonus_ = contact_lm_max_bonus;
  }

  lexicon_trie_ =
      std::make_shared<fl::lib::text::Trie>(am_tokens_.Size(), sil_id_);

  base_lexicon_entries_ =
      LoadLexicon(lexicon_path, output_words_, am_tokens_);
  if (contact_class_word_id_ >= 0) {
    for (const LexiconEntry& entry : base_lexicon_entries_) {
      if (entry.word == contact_class_word_) {
        throw std::runtime_error(
            "contact class word must not have a static lexicon pronunciation: " +
            contact_class_word_);
      }
    }
  }

  auto start_state = word_lm_->start(false);
  for (const LexiconEntry& entry : base_lexicon_entries_) {
    fl::lib::text::LMStatePtr ignored_state;
    float score = 0.0f;
    std::tie(ignored_state, score) = word_lm_->score(start_state, entry.word_id);
    lexicon_trie_->insert(entry.token_ids, entry.word_id, score);
    word_spellings_[entry.word].push_back(entry.token_ids);
  }
  lexicon_entry_count_ = static_cast<int>(base_lexicon_entries_.size());
  lexicon_trie_->smear(SmearingModeFromString(options_.smearing));
  output_mapper_ = OutputSequenceMapper::Load(mapping_path, output_words_);
}

FlashlightDecoderResource::FlashlightDecoderResource(
    const FlashlightDecoderResource& base,
    std::shared_ptr<fl::lib::text::Trie> combined_trie,
    fl::lib::text::LMPtr contextual_lm,
    std::shared_ptr<const DynamicContactLexicon> dynamic_contacts)
    : am_tokens_(base.am_tokens_),
      output_words_(base.output_words_),
      token_fl_dict_(base.token_fl_dict_),
      word_fl_dict_(base.word_fl_dict_),
      lexicon_trie_(std::move(combined_trie)),
      word_lm_(std::move(contextual_lm)),
      contact_word_lm_(base.contact_word_lm_),
      contact_lm_weight_(base.contact_lm_weight_),
      contact_lm_accumulation_factor_(base.contact_lm_accumulation_factor_),
      contact_lm_max_bonus_(base.contact_lm_max_bonus_),
      output_mapper_(base.output_mapper_),
      options_(base.options_),
      blank_id_(base.blank_id_),
      sil_id_(base.sil_id_),
      unk_word_id_(base.unk_word_id_),
      lexicon_entry_count_(base.lexicon_entry_count_),
      contact_class_word_(base.contact_class_word_),
      contact_class_word_id_(base.contact_class_word_id_),
      base_lexicon_entries_(base.base_lexicon_entries_),
      word_spellings_(base.word_spellings_),
      dynamic_contacts_(std::move(dynamic_contacts)) {
  // The contextual wrapper returns already weighted main-LM and contact-bias
  // contributions.  Keeping the external Flashlight multiplier at one avoids
  // coupling lm_weight with contact_lm_weight.
  options_.lm_weight = 1.0;
}

std::shared_ptr<const FlashlightDecoderResource>
FlashlightDecoderResource::CreateContactContextResource(
    const FlashlightDecoderResource& base,
    std::shared_ptr<fl::lib::text::Trie> combined_trie,
    fl::lib::text::LMPtr contextual_lm,
    std::shared_ptr<const DynamicContactLexicon> dynamic_contacts) {
  if (!base.SupportsRuntimeContacts()) {
    throw std::runtime_error(
        "contact decoder resource requires a contact-ready base resource");
  }
  if (!combined_trie || !contextual_lm || !dynamic_contacts) {
    throw std::runtime_error("invalid contact decoder resource components");
  }
  return std::shared_ptr<const FlashlightDecoderResource>(
      new FlashlightDecoderResource(base, std::move(combined_trie),
                                    std::move(contextual_lm),
                                    std::move(dynamic_contacts)));
}

}  // namespace asr_sdk::internal::flashlight_decoder
