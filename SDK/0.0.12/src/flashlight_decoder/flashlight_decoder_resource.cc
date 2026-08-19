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
#include "flashlight_decoder/max_fusion_lm.h"

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
    std::vector<FixedLmConfig> fixed_lms,
    const std::filesystem::path& pre_lm_mapping_path,
    const std::filesystem::path& final_mapping_path,
    FlashlightDecoderOptions options, std::string blank_token,
    std::string sil_token, std::string unk_word, double length_penalty)
    : am_tokens_(tokens_path),
      output_words_(words_path),
      token_fl_dict_(MakeTokenDictionary(am_tokens_)),
      word_fl_dict_(MakeWordDictionary(output_words_)),
      fixed_lms_(std::move(fixed_lms)),
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
  if (!std::isfinite(length_penalty) || length_penalty < 0.0) {
    throw std::runtime_error("length_penalty must be finite and >= 0");
  }
  options_.lm_weight = 1.0;
  options_.word_score = -length_penalty;
  auto pre_lm_mapper = std::make_shared<OutputSequenceMapper>(
      OutputSequenceMapper::Load(pre_lm_mapping_path, output_words_));
  pre_lm_mapper->ValidateForPreLm(pre_lm_mapping_path.empty()
                                     ? std::filesystem::path("output_mapping.txt")
                                     : pre_lm_mapping_path);
  pre_lm_mapper_ = pre_lm_mapper;
  final_output_mapper_ =
      OutputSequenceMapper::Load(final_mapping_path, output_words_);
  auto max_fusion_lm = std::make_shared<MaxFusionLm>(
      fixed_lms_, word_fl_dict_, nullptr, options_.word_score,
      pre_lm_mapper_);
  word_lm_ = max_fusion_lm;

  lexicon_trie_ =
      std::make_shared<fl::lib::text::Trie>(am_tokens_.Size(), sil_id_);

  base_lexicon_entries_ =
      LoadLexicon(lexicon_path, output_words_, am_tokens_);

  auto start_state = word_lm_->start(false);
  for (const LexiconEntry& entry : base_lexicon_entries_) {
    fl::lib::text::LMStatePtr ignored_state;
    float score = 0.0f;
    std::tie(ignored_state, score) =
        max_fusion_lm->ScoreForLexiconSmearing(start_state, entry.word_id);
    lexicon_trie_->insert(entry.token_ids, entry.word_id, score);
    word_spellings_[entry.word].push_back(entry.token_ids);
  }
  lexicon_entry_count_ = static_cast<int>(base_lexicon_entries_.size());
  lexicon_trie_->smear(SmearingModeFromString(options_.smearing));
}

FlashlightDecoderResource::FlashlightDecoderResource(
    const FlashlightDecoderResource& base,
    std::shared_ptr<fl::lib::text::Trie> combined_trie,
    std::shared_ptr<const DynamicContactLexicon> dynamic_contacts)
    : am_tokens_(base.am_tokens_),
      output_words_(base.output_words_),
      token_fl_dict_(base.token_fl_dict_),
      word_fl_dict_(base.word_fl_dict_),
      lexicon_trie_(std::move(combined_trie)),
      fixed_lms_(base.fixed_lms_),
      pre_lm_mapper_(base.pre_lm_mapper_),
      final_output_mapper_(base.final_output_mapper_),
      options_(base.options_),
      blank_id_(base.blank_id_),
      sil_id_(base.sil_id_),
      unk_word_id_(base.unk_word_id_),
      lexicon_entry_count_(base.lexicon_entry_count_),
      base_lexicon_entries_(base.base_lexicon_entries_),
      word_spellings_(base.word_spellings_),
      dynamic_contacts_(std::move(dynamic_contacts)) {
  word_lm_ = std::make_shared<MaxFusionLm>(
      fixed_lms_, word_fl_dict_, dynamic_contacts_, options_.word_score,
      pre_lm_mapper_);
  options_.lm_weight = 1.0;
}

bool FlashlightDecoderResource::SupportsRuntimeSlots() const {
  return std::any_of(fixed_lms_.begin(), fixed_lms_.end(),
                     [](const FixedLmConfig& lm) {
                       return lm.type == FixedLmType::kBias;
                     });
}

bool FlashlightDecoderResource::SupportsRuntimeSlot(
    const std::string& slot) const {
  for (const FixedLmConfig& lm : fixed_lms_) {
    if (lm.type == FixedLmType::kBias &&
        std::find(lm.slots.begin(), lm.slots.end(), slot) != lm.slots.end()) {
      return true;
    }
  }
  return false;
}

std::shared_ptr<const FlashlightDecoderResource>
FlashlightDecoderResource::CreateSlotContextResource(
    const FlashlightDecoderResource& base,
    std::shared_ptr<fl::lib::text::Trie> combined_trie,
    std::shared_ptr<const DynamicContactLexicon> dynamic_contacts) {
  if (!base.SupportsRuntimeSlots()) {
    throw std::runtime_error(
        "slot decoder resource requires a bias-LM-ready base resource");
  }
  if (!combined_trie || !dynamic_contacts) {
    throw std::runtime_error("invalid slot decoder resource components");
  }
  return std::shared_ptr<const FlashlightDecoderResource>(
      new FlashlightDecoderResource(base, std::move(combined_trie),
                                    std::move(dynamic_contacts)));
}

}  // namespace asr_sdk::internal::flashlight_decoder
