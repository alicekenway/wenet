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
    float word_score,
    std::shared_ptr<const OutputSequenceMapper> pre_lm_mapper)
    : dynamic_slots_(std::move(dynamic_slots)),
      pre_lm_mapper_(std::move(pre_lm_mapper)),
      word_score_(word_score) {
  auto fixed_lms = std::make_shared<SharedFixedLmResources>();
  fixed_lms->models.reserve(configs.size());
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
    fixed_lms->models.push_back(std::move(loaded));
  }
  if (fixed_lms->models.empty()) {
    throw std::invalid_argument("max-fusion LM requires at least one model");
  }
  fixed_lms_ = std::move(fixed_lms);
  if (pre_lm_mapper_ && !pre_lm_mapper_->empty()) {
    pre_lm_mapper_->ValidateForPreLm("output_mapping.txt");
  }
}

MaxFusionLm::MaxFusionLm(
    std::shared_ptr<const SharedFixedLmResources> fixed_lms,
    std::shared_ptr<const DynamicContactLexicon> dynamic_slots,
    float word_score,
    std::shared_ptr<const OutputSequenceMapper> pre_lm_mapper)
    : fixed_lms_(std::move(fixed_lms)),
      dynamic_slots_(std::move(dynamic_slots)),
      pre_lm_mapper_(std::move(pre_lm_mapper)),
      word_score_(word_score) {
  if (!fixed_lms_ || fixed_lms_->models.empty()) {
    throw std::invalid_argument("max-fusion LM requires fixed resources");
  }
  if (pre_lm_mapper_ && !pre_lm_mapper_->empty()) {
    pre_lm_mapper_->ValidateForPreLm("output_mapping.txt");
  }
}

fl::lib::text::LMStatePtr MaxFusionLm::start(bool start_with_nothing) {
  auto state = std::make_shared<MaxFusionLmState>();
  state->model_states.reserve(Models().size());
  for (const LoadedFixedLm& model : Models()) {
    state->model_states.push_back(model.lm->start(start_with_nothing));
  }
  return state;
}

std::shared_ptr<MaxFusionLmState> MaxFusionLm::RequireState(
    const fl::lib::text::LMStatePtr& state) const {
  auto typed = std::static_pointer_cast<MaxFusionLmState>(state);
  if (!typed || typed->model_states.size() != Models().size()) {
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

float MaxFusionLm::ScoreOne(
    const std::vector<fl::lib::text::LMStatePtr>& model_states,
    int user_token_id, MaxFusionLmState* out) const {
  const bool is_dynamic =
      dynamic_slots_ && dynamic_slots_->IsDynamicContactId(user_token_id);
  const RuntimeContactForm* form =
      is_dynamic ? &dynamic_slots_->ContactFormForId(user_token_id) : nullptr;

  out->model_states.resize(Models().size());
  out->winner_filename.clear();
  out->winner_type.clear();
  out->raw_score = 0.0f;
  out->reference_score = 0.0f;
  out->adjusted_score = 0.0f;
  out->weighted_score = 0.0f;
  float best = NegativeInfinity();
  bool has_candidate = false;
  for (size_t index = 0; index < Models().size(); ++index) {
    const LoadedFixedLm& model = Models()[index];
    const auto& model_state = model_states[index];
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
    if (!has_candidate) return NegativeInfinity();
    best += ContactWordScoreCorrection(form->logical_word_count, word_score_);
  } else if (!has_candidate) {
    best = 0.0f;
  }
  return best;
}

std::pair<fl::lib::text::LMStatePtr, float>
MaxFusionLm::ScoreForLexiconSmearing(
    const fl::lib::text::LMStatePtr& state, int user_token_id) {
  const auto in = RequireState(state);
  auto out = std::make_shared<MaxFusionLmState>();
  const float contribution = ScoreOne(in->model_states, user_token_id,
                                      out.get());
  return std::make_pair(std::move(out), contribution);
}

std::pair<fl::lib::text::LMStatePtr, float> MaxFusionLm::score(
    const fl::lib::text::LMStatePtr& state, int user_token_id) {
  const auto in = RequireState(state);
  auto out = in->child<MaxFusionLmState>(user_token_id);
  const float source_contribution =
      ScoreOne(in->model_states, user_token_id, out.get());
  out->raw_word_count = in->raw_word_count + 1;

  if (!pre_lm_mapper_ || pre_lm_mapper_->empty()) {
    out->recent_raw_events.clear();
    return std::make_pair(std::move(out), source_contribution);
  }

  out->recent_raw_events = in->recent_raw_events;
  out->recent_raw_events.push_back(
      RecentRawLmEvent{user_token_id, in->model_states,
                       source_contribution});
  std::vector<int> recent_ids;
  recent_ids.reserve(out->recent_raw_events.size());
  for (const RecentRawLmEvent& event : out->recent_raw_events) {
    recent_ids.push_back(event.word_id);
  }
  const auto match = pre_lm_mapper_->MatchIdSuffix(
      recent_ids, out->raw_word_count);
  if (!match.has_value()) {
    const size_t keep = pre_lm_mapper_->MaxIdSourceLength() > 0
                            ? pre_lm_mapper_->MaxIdSourceLength() - 1
                            : 0;
    if (out->recent_raw_events.size() > keep) {
      out->recent_raw_events.erase(
          out->recent_raw_events.begin(),
          out->recent_raw_events.end() - static_cast<std::ptrdiff_t>(keep));
    }
    return std::make_pair(std::move(out), source_contribution);
  }

  const size_t source_length = match->source_length;
  const size_t source_begin = out->recent_raw_events.size() - source_length;
  std::vector<fl::lib::text::LMStatePtr> replay_states =
      out->recent_raw_events[source_begin].model_states_before;
  float prior_source_contribution = 0.0f;
  for (size_t index = source_begin;
       index + 1 < out->recent_raw_events.size(); ++index) {
    prior_source_contribution +=
        out->recent_raw_events[index].contribution;
  }

  float mapped_contribution = 0.0f;
  MaxFusionLmState replay;
  for (int target_word_id : *match->target) {
    const float contribution =
        ScoreOne(replay_states, target_word_id, &replay);
    mapped_contribution += contribution;
    replay_states = replay.model_states;
  }
  out->model_states = std::move(replay_states);
  out->recent_raw_events.clear();
  const float length_correction =
      word_score_ * (static_cast<float>(match->target->size()) -
                     static_cast<float>(source_length));
  const float corrected_contribution =
      mapped_contribution - prior_source_contribution + length_correction;
  out->winner_filename = "output_mapping.txt";
  out->winner_type = "pre_lm_mapping";
  out->raw_score = prior_source_contribution + source_contribution;
  out->reference_score = mapped_contribution;
  out->adjusted_score = mapped_contribution - out->raw_score;
  out->weighted_score = corrected_contribution;
  return std::make_pair(std::move(out), corrected_contribution);
}

std::pair<fl::lib::text::LMStatePtr, float> MaxFusionLm::finish(
    const fl::lib::text::LMStatePtr& state) {
  const auto in = RequireState(state);
  auto out = in->child<MaxFusionLmState>(-1);
  out->model_states.resize(Models().size());
  float best = NegativeInfinity();
  bool has_ngram = false;
  for (size_t index = 0; index < Models().size(); ++index) {
    const LoadedFixedLm& model = Models()[index];
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
  for (size_t index = 0; index < Models().size(); ++index) {
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
      Models()[index].lm->updateCache(std::move(model_states));
    }
  }
}

}  // namespace asr_sdk::internal::flashlight_decoder
