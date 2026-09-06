#include "flashlight_decoder/flashlight_decoder_resource.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <mutex>
#include <set>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "flashlight/lib/text/decoder/lm/KenLM.h"
#include "flashlight_decoder/contact_bias.h"
#include "flashlight_decoder/lexicon_loader.h"
#include "flashlight_decoder/max_fusion_lm.h"
#include "flashlight_decoder/word_spelling_generator.h"

namespace asr_sdk::internal::flashlight_decoder {
namespace {

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

// One immutable owner for all package-sized decoder data. Runtime contexts
// retain this object instead of copying dictionaries, lexicon metadata, fixed
// KenLMs, and the compact base lexicon. The legacy fields remain empty in an
// SDK 0.0.16 runtime process and are used only by the host compiler.
struct SharedDecoderResources {
  SharedDecoderResources(const std::filesystem::path& tokens_path,
                         const std::filesystem::path& words_path)
      : am_tokens(tokens_path), output_words(words_path) {}

  sherpa_onnx_wenet::TokenTable am_tokens;
  WordDictionary output_words;
  std::shared_ptr<const CompactLexicon> compact_lexicon;
  std::shared_ptr<fl::lib::text::Trie> lexicon_trie;
  std::shared_ptr<const SharedFixedLmResources> fixed_lm_resources;
  std::vector<FixedLmConfig> fixed_lms;
  std::shared_ptr<const OutputSequenceMapper> pre_lm_mapper;
  OutputSequenceMapper final_output_mapper;
  FlashlightDecoderOptions options;
  int blank_id = 0;
  int sil_id = 0;
  int unk_word_id = 0;
  int lexicon_entry_count = 0;
  std::vector<LexiconEntry> base_lexicon_entries;
  std::vector<float> base_word_scores;
  std::unordered_map<std::string, std::vector<std::vector<int>>>
      word_spellings;
  mutable std::once_flag runtime_lookahead_once;
  std::filesystem::path sentencepiece_model;
  mutable std::once_flag word_speller_once;
  mutable std::unique_ptr<WordSpellingGenerator> word_speller;
  mutable std::shared_ptr<const RuntimeTrieLookahead> runtime_lookahead;

  std::shared_ptr<const RuntimeTrieLookahead> GetRuntimeLookahead(
      const fl::lib::text::LMPtr& lm) const;
};

std::shared_ptr<const RuntimeTrieLookahead>
SharedDecoderResources::GetRuntimeLookahead(
    const fl::lib::text::LMPtr& lm) const {
  std::call_once(runtime_lookahead_once, [&]() {
    auto out = std::make_shared<RuntimeTrieLookahead>();
    std::vector<float> word_scores(output_words.Size(), 0.0f);
    std::vector<bool> scored(output_words.Size(), false);
    const auto start_state = lm->start(false);
    for (const LexiconEntry& entry : base_lexicon_entries) {
      if (scored[entry.word_id]) continue;
      fl::lib::text::LMStatePtr ignored;
      std::tie(ignored, word_scores[entry.word_id]) =
          lm->score(start_state, entry.word_id);
      scored[entry.word_id] = true;
    }

    // Only homophone terminals and pre-LM-mapped words can differ from the
    // base trie's legacy lookahead. Mark their ancestor paths, then recompute
    // just that sparse subgraph bottom-up instead of walking every trie node.
    std::unordered_map<const fl::lib::text::TrieNode*, float>
        dirty_terminal_scores;
    std::unordered_map<const fl::lib::text::TrieNode*, int> dirty_depths;
    dirty_depths.emplace(lexicon_trie->getRoot(), 0);
    for (const LexiconEntry& entry : base_lexicon_entries) {
      const fl::lib::text::TrieNode* node = lexicon_trie->getRoot();
      std::vector<const fl::lib::text::TrieNode*> path{node};
      for (int token : entry.token_ids) {
        const auto child = node->children.find(token);
        if (child == node->children.end()) {
          node = nullptr;
          break;
        }
        node = child->second.get();
        path.push_back(node);
      }
      if (!node) {
        throw std::runtime_error(
            "base lexicon entry disappeared while preparing runtime lookahead");
      }
      const bool dirty_terminal =
          node->labels.size() > 1 ||
          word_scores[entry.word_id] != base_word_scores[entry.word_id];
      if (!dirty_terminal) continue;
      auto [terminal, inserted] = dirty_terminal_scores.emplace(
          node, word_scores[entry.word_id]);
      if (!inserted) {
        terminal->second = std::max(
            terminal->second, word_scores[entry.word_id]);
      }
      for (size_t depth = 0; depth < path.size(); ++depth) {
        dirty_depths.emplace(path[depth], static_cast<int>(depth));
      }
    }

    std::vector<std::pair<const fl::lib::text::TrieNode*, int>> dirty_nodes(
        dirty_depths.begin(), dirty_depths.end());
    std::sort(dirty_nodes.begin(), dirty_nodes.end(),
              [](const auto& left, const auto& right) {
                return left.second > right.second;
              });
    for (const auto& [node, ignored_depth] : dirty_nodes) {
      float strict_score = -std::numeric_limits<float>::infinity();
      const auto terminal = dirty_terminal_scores.find(node);
      if (terminal != dirty_terminal_scores.end()) {
        strict_score = terminal->second;
      } else if (!node->scores.empty()) {
        // A non-dirty terminal has one label and no mapping, so its strict
        // terminal score is exactly the stored label score.
        strict_score = node->scores.front();
      }
      for (const auto& child : node->children) {
        const auto child_override = out->find(child.second.get());
        strict_score = std::max(
            strict_score, child_override == out->end()
                              ? child.second->maxScore
                              : child_override->second);
      }
      if (strict_score != node->maxScore) out->emplace(node, strict_score);
    }
    runtime_lookahead = std::move(out);
  });
  return runtime_lookahead;
}

FlashlightDecoderResource::FlashlightDecoderResource(
    const std::filesystem::path& tokens_path,
    const std::filesystem::path& words_path,
    const std::filesystem::path& lexicon_path,
    std::vector<FixedLmConfig> fixed_lms,
    const std::filesystem::path& pre_lm_mapping_path,
    const std::filesystem::path& final_mapping_path,
    FlashlightDecoderOptions options, std::string blank_token,
    std::string sil_token, std::string unk_word, double length_penalty,
    bool compact_binary, uint64_t expected_dependency_hash,
    const std::filesystem::path& sentencepiece_model) {
  auto shared =
      std::make_shared<SharedDecoderResources>(tokens_path, words_path);
  shared->fixed_lms = std::move(fixed_lms);
  shared->sentencepiece_model = sentencepiece_model;
  shared->options = std::move(options);
  if (!shared->am_tokens.Contains(blank_token)) {
    throw std::runtime_error("blank token not found in tokens.txt: " +
                             blank_token);
  }
  if (!shared->am_tokens.Contains(sil_token)) {
    throw std::runtime_error("silence/separator token not found in tokens.txt: " +
                             sil_token);
  }
  if (!shared->output_words.Contains(unk_word)) {
    throw std::runtime_error("unknown word not found in words.txt: " +
                             unk_word);
  }
  std::set<std::string> runtime_slots;
  for (const FixedLmConfig& lm : shared->fixed_lms) {
    for (const std::string& slot : lm.slots) {
      runtime_slots.insert(slot);
      if (!shared->output_words.Contains(slot) || slot == unk_word ||
          shared->am_tokens.Contains(slot)) {
        throw std::runtime_error("invalid bias slot token placement: " + slot);
      }
    }
  }
  shared->blank_id = shared->am_tokens.Id(blank_token);
  shared->sil_id = shared->am_tokens.Id(sil_token);
  shared->unk_word_id = shared->output_words.Id(unk_word);
  if (!std::isfinite(length_penalty) || length_penalty < 0.0) {
    throw std::runtime_error("length_penalty must be finite and >= 0");
  }
  shared->options.lm_weight = 1.0;
  shared->options.word_score = -length_penalty;
  auto pre_lm_mapper = std::make_shared<OutputSequenceMapper>(
      OutputSequenceMapper::Load(pre_lm_mapping_path, shared->output_words));
  pre_lm_mapper->ValidateForPreLm(pre_lm_mapping_path.empty()
                                     ? std::filesystem::path("output_mapping.txt")
                                     : pre_lm_mapping_path);
  shared->pre_lm_mapper = pre_lm_mapper;
  shared->final_output_mapper =
      OutputSequenceMapper::Load(final_mapping_path, shared->output_words);
  const fl::lib::text::Dictionary word_dictionary =
      MakeWordDictionary(shared->output_words);
  auto max_fusion_lm = std::make_shared<MaxFusionLm>(
      shared->fixed_lms, word_dictionary, nullptr,
      shared->options.word_score, shared->pre_lm_mapper);
  shared->fixed_lm_resources = max_fusion_lm->FixedResources();
  for (const LoadedFixedLm& loaded : shared->fixed_lm_resources->models) {
    for (const std::string& slot : loaded.config.slots) {
      if (!loaded.lm->HasWord(slot)) {
        throw std::runtime_error("LM " + loaded.config.filename +
                                 " is missing slot token " + slot);
      }
    }
  }
  word_lm_ = max_fusion_lm;

  if (compact_binary) {
    shared->compact_lexicon =
        CompactLexicon::Load(lexicon_path, expected_dependency_hash);
    if (shared->compact_lexicon->TokenCount() !=
            static_cast<uint32_t>(shared->am_tokens.ModelVocabSize()) ||
        shared->compact_lexicon->WordCount() !=
            static_cast<uint32_t>(shared->output_words.Size()) ||
        shared->compact_lexicon->BlankId() != shared->blank_id ||
        shared->compact_lexicon->SilenceId() != shared->sil_id ||
        shared->compact_lexicon->UnknownWordId() != shared->unk_word_id) {
      throw std::runtime_error(
          "compact lexicon metadata does not match tokens.txt/words.txt");
    }
    shared->lexicon_entry_count =
        static_cast<int>(shared->compact_lexicon->LexiconEntryCount());
    // The static trie is entirely memory-mapped. Only this tiny context trie
    // remains heap allocated for the no-contact decoder.
    overlay_trie_ = std::make_shared<fl::lib::text::Trie>(
        shared->am_tokens.Size(), shared->sil_id);
  } else {
    // This path is used by the host compiler and legacy parity tests only.
    shared->lexicon_trie = std::make_shared<fl::lib::text::Trie>(
        shared->am_tokens.Size(), shared->sil_id);
    shared->base_lexicon_entries = LoadLexicon(
        lexicon_path, shared->output_words, shared->am_tokens);
    for (const LexiconEntry& entry : shared->base_lexicon_entries) {
      if (runtime_slots.find(entry.word) != runtime_slots.end()) {
        throw std::runtime_error(
            "slot token must not have a static pronunciation: " + entry.word);
      }
    }

    auto start_state = word_lm_->start(false);
    shared->base_word_scores.assign(shared->output_words.Size(), 0.0f);
    std::vector<bool> scored(shared->output_words.Size(), false);
    for (const LexiconEntry& entry : shared->base_lexicon_entries) {
      if (!scored[entry.word_id]) {
        fl::lib::text::LMStatePtr ignored_state;
        std::tie(ignored_state, shared->base_word_scores[entry.word_id]) =
            max_fusion_lm->ScoreForLexiconSmearing(start_state, entry.word_id);
        scored[entry.word_id] = true;
      }
      shared->lexicon_trie->insert(
          entry.token_ids, entry.word_id,
          shared->base_word_scores[entry.word_id]);
      shared->word_spellings[entry.word].push_back(entry.token_ids);
    }
    shared->lexicon_entry_count =
        static_cast<int>(shared->base_lexicon_entries.size());
    shared->lexicon_trie->smear(
        SmearingModeFromString(shared->options.smearing));
  }
  shared_ = std::move(shared);
}

FlashlightDecoderResource::FlashlightDecoderResource(
    const FlashlightDecoderResource& base,
    std::shared_ptr<fl::lib::text::Trie> overlay_trie,
    std::shared_ptr<const DynamicContactLexicon> dynamic_contacts)
    : shared_(base.shared_),
      overlay_trie_(std::move(overlay_trie)),
      dynamic_contacts_(std::move(dynamic_contacts)) {
  word_lm_ = std::make_shared<MaxFusionLm>(
      shared_->fixed_lm_resources, dynamic_contacts_,
      shared_->options.word_score, shared_->pre_lm_mapper);
}

const sherpa_onnx_wenet::TokenTable& FlashlightDecoderResource::AmTokens()
    const { return shared_->am_tokens; }
const WordDictionary& FlashlightDecoderResource::OutputWords() const {
  return shared_->output_words;
}
const std::shared_ptr<const CompactLexicon>&
FlashlightDecoderResource::Lexicon() const {
  return shared_->compact_lexicon;
}
const std::shared_ptr<fl::lib::text::Trie>&
FlashlightDecoderResource::LexiconTrie() const {
  return shared_->lexicon_trie;
}
const std::vector<FixedLmConfig>& FlashlightDecoderResource::FixedLms() const {
  return shared_->fixed_lms;
}
const OutputSequenceMapper& FlashlightDecoderResource::PreLmMapper() const {
  return *shared_->pre_lm_mapper;
}
const OutputSequenceMapper& FlashlightDecoderResource::FinalMapper() const {
  return shared_->final_output_mapper;
}
const FlashlightDecoderOptions& FlashlightDecoderResource::Options() const {
  return shared_->options;
}
int FlashlightDecoderResource::BlankId() const { return shared_->blank_id; }
int FlashlightDecoderResource::SilenceId() const { return shared_->sil_id; }
int FlashlightDecoderResource::UnknownWordId() const {
  return shared_->unk_word_id;
}
int FlashlightDecoderResource::LexiconEntryCount() const {
  return shared_->lexicon_entry_count;
}
int FlashlightDecoderResource::BaseWordCount() const {
  return shared_->output_words.Size();
}
const std::vector<LexiconEntry>&
FlashlightDecoderResource::BaseLexiconEntries() const {
  return shared_->base_lexicon_entries;
}
const WordSpellingGenerator& FlashlightDecoderResource::ContactWordSpeller() const {
  std::call_once(shared_->word_speller_once, [&]() {
    shared_->word_speller = std::make_unique<WordSpellingGenerator>(
        shared_->am_tokens, shared_->blank_id,
        shared_->am_tokens.Token(shared_->sil_id), shared_->sentencepiece_model);
  });
  return *shared_->word_speller;
}

std::vector<std::vector<int>> FlashlightDecoderResource::WordSpellings(
    const std::string& word) const {
  if (!shared_->output_words.Contains(word)) return {};
  if (shared_->compact_lexicon) {
    return shared_->compact_lexicon->WordSpellings(
        shared_->output_words.Id(word));
  }
  const auto found = shared_->word_spellings.find(word);
  return found == shared_->word_spellings.end()
             ? std::vector<std::vector<int>>()
             : found->second;
}

std::shared_ptr<const RuntimeTrieLookahead>
FlashlightDecoderResource::LegacyContextLookaheadForCompiler() const {
  if (!shared_->lexicon_trie) {
    throw std::runtime_error(
        "legacy lookahead requested from compact runtime resource");
  }
  return shared_->GetRuntimeLookahead(word_lm_);
}

bool FlashlightDecoderResource::SupportsRuntimeSlots() const {
  return std::any_of(shared_->fixed_lms.begin(), shared_->fixed_lms.end(),
                     [](const FixedLmConfig& lm) {
                       return lm.type == FixedLmType::kBias;
                     });
}

bool FlashlightDecoderResource::SupportsRuntimeSlot(
    const std::string& slot) const {
  for (const FixedLmConfig& lm : shared_->fixed_lms) {
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
    std::shared_ptr<fl::lib::text::Trie> overlay_trie,
    std::shared_ptr<const DynamicContactLexicon> dynamic_contacts) {
  if (!base.SupportsRuntimeSlots()) {
    throw std::runtime_error(
        "slot decoder resource requires a bias-LM-ready base resource");
  }
  if (!overlay_trie || !dynamic_contacts) {
    throw std::runtime_error("invalid slot decoder resource components");
  }
  return std::shared_ptr<const FlashlightDecoderResource>(
      new FlashlightDecoderResource(base, std::move(overlay_trie),
                                    std::move(dynamic_contacts)));
}

}  // namespace asr_sdk::internal::flashlight_decoder
