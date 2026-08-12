#include "flashlight_decoder/max_fusion_lm.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "flashlight_decoder/contact_bias.h"
#include "flashlight_decoder/dynamic_contact_lexicon.h"

namespace asr_sdk::internal::flashlight_decoder {
namespace {

float NegativeInfinity() {
  return -std::numeric_limits<float>::infinity();
}

}  // namespace

MaxFusionLm::MaxFusionLm(
    std::vector<FixedLmConfig> configs,
    const fl::lib::text::Dictionary& words,
    std::shared_ptr<const DynamicContactLexicon> dynamic_slots,
    float word_score)
    : dynamic_slots_(std::move(dynamic_slots)), word_score_(word_score) {
  models_.reserve(configs.size());
  for (FixedLmConfig& config : configs) {
    LoadedFixedLm loaded;
    loaded.config = std::move(config);
    loaded.lm = std::make_shared<fl::lib::text::KenLM>(
        loaded.config.path.string(), words);
    for (const std::string& slot : loaded.config.slots) {
      if (!words.contains(slot)) {
        throw std::runtime_error("slot token is absent from words.txt: " + slot);
      }
      loaded.slot_word_ids.insert(words.getIndex(slot));
    }
    models_.push_back(std::move(loaded));
  }
  if (models_.empty()) {
    throw std::invalid_argument("max-fusion LM requires at least one model");
  }
}

fl::lib::text::LMStatePtr MaxFusionLm::start(bool start_with_nothing) {
  auto state = std::make_shared<MaxFusionLmState>();
  state->model_states.reserve(models_.size());
  for (const LoadedFixedLm& model : models_) {
    state->model_states.push_back(model.lm->start(start_with_nothing));
  }
  return state;
}

std::shared_ptr<MaxFusionLmState> MaxFusionLm::RequireState(
    const fl::lib::text::LMStatePtr& state) const {
  auto typed = std::static_pointer_cast<MaxFusionLmState>(state);
  if (!typed || typed->model_states.size() != models_.size()) {
    throw std::invalid_argument("invalid max-fusion LM state");
  }
  return typed;
}

float MaxFusionLm::NgramContribution(
    const LoadedFixedLm& model,
    const fl::lib::text::LMStatePtr& state,
    int word_id,
    fl::lib::text::LMStatePtr* next,
    float* raw, float* reference, float* adjusted) const {
  float word_score = 0.0f;
  if (model.lm->HasUserToken(word_id)) {
    std::tie(*next, word_score) = model.lm->score(state, word_id);
  } else {
    std::tie(*next, word_score) = model.lm->scoreUnknown(state, word_id);
  }
  fl::lib::text::LMStatePtr ignored;
  float unknown_score = 0.0f;
  std::tie(ignored, unknown_score) = model.lm->scoreUnknown(state);
  float relative = word_score - unknown_score;
  *raw = word_score;
  *reference = unknown_score;
  if (model.config.clip) {
    relative = std::clamp(relative,
                          static_cast<float>(model.config.clip_lower),
                          static_cast<float>(model.config.clip_upper));
  }
  *adjusted = relative;
  return static_cast<float>(model.config.weight) * relative;
}

std::pair<fl::lib::text::LMStatePtr, float> MaxFusionLm::score(
    const fl::lib::text::LMStatePtr& state, int user_token_id) {
  const auto in = RequireState(state);
  const bool is_dynamic =
      dynamic_slots_ && dynamic_slots_->IsDynamicContactId(user_token_id);
  const RuntimeContactForm* form =
      is_dynamic ? &dynamic_slots_->ContactFormForId(user_token_id) : nullptr;

  auto out = in->child<MaxFusionLmState>(user_token_id);
  out->model_states.resize(models_.size());
  float best = NegativeInfinity();
  bool has_candidate = false;
  for (size_t index = 0; index < models_.size(); ++index) {
    const LoadedFixedLm& model = models_[index];
    const auto& model_state = in->model_states[index];
    if (model.config.type == FixedLmType::kNgram) {
      const int observed_word = is_dynamic ? form->slot_word_id : user_token_id;
      if (is_dynamic) {
        float ignored_score = 0.0f;
        if (model.lm->HasUserToken(observed_word)) {
          std::tie(out->model_states[index], ignored_score) =
              model.lm->score(model_state, observed_word);
        } else {
          std::tie(out->model_states[index], ignored_score) =
              model.lm->scoreUnknown(model_state, observed_word);
        }
      } else {
        float raw = 0.0f, reference = 0.0f, adjusted = 0.0f;
        const float contribution = NgramContribution(
            model, model_state, observed_word, &out->model_states[index],
            &raw, &reference, &adjusted);
        if (!has_candidate || contribution > best) {
          best = contribution;
          out->winner_filename = model.config.filename;
          out->winner_type = "ngram";
          out->raw_score = raw;
          out->reference_score = reference;
          out->adjusted_score = adjusted;
          out->weighted_score = contribution;
        }
        has_candidate = true;
      }
      continue;
    }

    const int observed_word = is_dynamic ? form->slot_word_id : user_token_id;
    const auto matched = model.lm->scoreWithMetadata(model_state, observed_word);
    out->model_states[index] = matched.state;
    if (is_dynamic &&
        model.slot_word_ids.find(observed_word) != model.slot_word_ids.end() &&
        matched.score < 0.0f && matched.ngram_length >= 2) {
      const float contribution =
          static_cast<float>(model.config.weight) * -matched.score *
          ContactBonusMultiplier(form->logical_word_count,
                                 model.config.accumulation_factor);
      if (!has_candidate || contribution > best) {
        best = contribution;
        out->winner_filename = model.config.filename;
        out->winner_type = "bias";
        out->raw_score = matched.score;
        out->reference_score = 0.0f;
        out->adjusted_score = -matched.score * ContactBonusMultiplier(
            form->logical_word_count, model.config.accumulation_factor);
        out->weighted_score = contribution;
      }
      has_candidate = true;
    }
  }

  if (is_dynamic) {
    if (!has_candidate) return std::make_pair(state, NegativeInfinity());
    best += ContactWordScoreCorrection(form->logical_word_count, word_score_);
  } else if (!has_candidate) {
    best = 0.0f;
  }
  return std::make_pair(std::move(out), best);
}

std::pair<fl::lib::text::LMStatePtr, float> MaxFusionLm::finish(
    const fl::lib::text::LMStatePtr& state) {
  const auto in = RequireState(state);
  auto out = in->child<MaxFusionLmState>(-1);
  out->model_states.resize(models_.size());
  float best = NegativeInfinity();
  bool has_ngram = false;
  for (size_t index = 0; index < models_.size(); ++index) {
    const LoadedFixedLm& model = models_[index];
    float eos_score = 0.0f;
    std::tie(out->model_states[index], eos_score) =
        model.lm->finish(in->model_states[index]);
    if (model.config.type != FixedLmType::kNgram) continue;
    fl::lib::text::LMStatePtr ignored;
    float unknown_score = 0.0f;
    std::tie(ignored, unknown_score) =
        model.lm->scoreUnknown(in->model_states[index]);
    float relative = eos_score - unknown_score;
    if (model.config.clip) {
      relative = std::clamp(relative,
                            static_cast<float>(model.config.clip_lower),
                            static_cast<float>(model.config.clip_upper));
    }
    const float contribution =
        static_cast<float>(model.config.weight) * relative;
    if (!has_ngram || contribution > best) {
      best = contribution;
      out->winner_filename = model.config.filename;
      out->winner_type = "ngram_eos";
      out->raw_score = eos_score;
      out->reference_score = unknown_score;
      out->adjusted_score = relative;
      out->weighted_score = contribution;
    }
    has_ngram = true;
  }
  return std::make_pair(std::move(out), has_ngram ? best : 0.0f);
}

void MaxFusionLm::updateCache(
    std::vector<fl::lib::text::LMStatePtr> states) {
  for (size_t index = 0; index < models_.size(); ++index) {
    std::vector<fl::lib::text::LMStatePtr> model_states;
    std::unordered_set<const fl::lib::text::LMState*> seen;
    for (const auto& state : states) {
      const auto typed = std::static_pointer_cast<MaxFusionLmState>(state);
      if (typed && index < typed->model_states.size() &&
          seen.insert(typed->model_states[index].get()).second) {
        model_states.push_back(typed->model_states[index]);
      }
    }
    if (!model_states.empty()) {
      models_[index].lm->updateCache(std::move(model_states));
    }
  }
}

}  // namespace asr_sdk::internal::flashlight_decoder
