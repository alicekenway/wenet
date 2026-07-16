#include "flashlight_decoder/contextual_contact_lm.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "flashlight_decoder/contact_bias.h"

namespace asr_sdk::internal::flashlight_decoder {

ContextualContactLm::ContextualContactLm(
    fl::lib::text::LMPtr main_lm, fl::lib::text::LMPtr contact_lm,
    std::shared_ptr<const DynamicContactLexicon> contacts,
    int contact_class_word_id, double main_lm_weight, double contact_lm_weight,
    double accumulation_factor, float word_score)
    : main_lm_(std::move(main_lm)),
      contact_lm_(std::move(contact_lm)),
      contacts_(std::move(contacts)),
      contact_class_word_id_(contact_class_word_id),
      main_lm_weight_(main_lm_weight),
      contact_lm_weight_(contact_lm_weight),
      accumulation_factor_(accumulation_factor),
      word_score_(word_score) {
  if (!main_lm_ || !contact_lm_ || !contacts_) {
    throw std::invalid_argument(
        "contextual contact LM requires main LM, contact LM, and contacts");
  }
  if (contact_class_word_id_ < 0) {
    throw std::invalid_argument("contextual contact LM has no contact class word");
  }
  if (!std::isfinite(main_lm_weight_)) {
    throw std::invalid_argument(
        "contextual contact LM requires finite main lm_weight");
  }
  if (!std::isfinite(contact_lm_weight_) || contact_lm_weight_ <= 0.0) {
    throw std::invalid_argument(
        "contextual contact LM requires finite positive contact_lm_weight");
  }
  if (!IsValidContactAccumulationFactor(accumulation_factor_)) {
    throw std::invalid_argument(
        "contextual contact LM accumulation factor must be between zero and one");
  }
  main_kenlm_ = std::dynamic_pointer_cast<fl::lib::text::KenLM>(main_lm_);
  contact_kenlm_ =
      std::dynamic_pointer_cast<fl::lib::text::KenLM>(contact_lm_);
  if (!main_kenlm_ || !contact_kenlm_) {
    throw std::invalid_argument(
        "contextual contact LM requires KenLM-backed main and contact LMs");
  }
}

fl::lib::text::LMStatePtr ContextualContactLm::start(bool start_with_nothing) {
  auto state = std::make_shared<ContextualContactLmState>();
  state->main_lm_state = main_lm_->start(start_with_nothing);
  state->contact_lm_state = contact_lm_->start(start_with_nothing);
  return state;
}

std::shared_ptr<ContextualContactLmState> ContextualContactLm::RequireState(
    const fl::lib::text::LMStatePtr& state) const {
  auto typed = std::static_pointer_cast<ContextualContactLmState>(state);
  if (!typed || !typed->main_lm_state || !typed->contact_lm_state) {
    throw std::invalid_argument("invalid contextual contact LM state");
  }
  return typed;
}

float ContextualContactLm::NegativeInfinity() {
  return -std::numeric_limits<float>::infinity();
}

std::pair<fl::lib::text::LMStatePtr, float> ContextualContactLm::score(
    const fl::lib::text::LMStatePtr& state, int user_token_id) {
  const auto in = RequireState(state);
  const bool is_contact = contacts_->IsDynamicContactId(user_token_id);

  fl::lib::text::LMStatePtr next_main;
  fl::lib::text::LMStatePtr next_contact;
  float combined_score = 0.0f;
  bool saw_base_word_before_contact = in->saw_base_word_before_contact;
  bool emitted_contact = in->emitted_contact;
  bool exact_standalone_contact = in->exact_standalone_contact;

  if (is_contact) {
    if (emitted_contact) {
      return std::make_pair(state, NegativeInfinity());
    }
    const RuntimeContactForm& form =
        contacts_->ContactFormForId(user_token_id);
    const auto matched = contact_kenlm_->scoreWithMetadata(
        in->contact_lm_state, contact_class_word_id_);
    next_contact = matched.state;
    // The handmade ARPA stores a terminal bonus S as -S.  A zero score means
    // that KenLM backed off to a scaffold n-gram, so this contact is not in a
    // configured context and the contact-only hypothesis must not survive.
    if (matched.score >= 0.0f || matched.ngram_length < 2) {
      return std::make_pair(state, NegativeInfinity());
    }
    const float bonus = -matched.score;
    combined_score = static_cast<float>(contact_lm_weight_) * bonus *
                     ContactBonusMultiplier(form.logical_word_count,
                                            accumulation_factor_);

    // Flashlight applies word_score once per lexicon label.  A dynamic label
    // represents the complete spoken form, so compensate for every logical
    // word after the first.  This is ordinary decoder bookkeeping, not a
    // contact preference: all contact forms with the same logical length get
    // the same correction and the final total matches literal word decoding.
    const float correction =
        ContactWordScoreCorrection(form.logical_word_count, word_score_);
    combined_score += correction;
    // The contact span intentionally receives no main-LM name score.  Reset
    // the main history so the following word is not incorrectly connected to
    // the word before the skipped runtime name.
    next_main = main_lm_->start(true);
    emitted_contact = true;
    // <CONTACT> by itself is represented as <s> <CONTACT> in the generated
    // ARPA.  It is valid only if no base word preceded it and no later base
    // word is emitted.
    exact_standalone_contact =
        !saw_base_word_before_contact && matched.ngram_length == 2;
  } else {
    if (user_token_id < 0 || user_token_id >= contacts_->BaseWordCount()) {
      throw std::out_of_range("unknown word id in contextual contact LM: " +
                              std::to_string(user_token_id));
    }
    if (exact_standalone_contact) {
      return std::make_pair(state, NegativeInfinity());
    }
    if (main_kenlm_->HasUserToken(user_token_id)) {
      float raw_main_score = 0.0f;
      std::tie(next_main, raw_main_score) =
          main_lm_->score(in->main_lm_state, user_token_id);
      combined_score =
          static_cast<float>(main_lm_weight_) * raw_main_score;
    } else {
      // Missing vocabulary is observable in debug output through the zero LM
      // contribution, but does not discard the AM hypothesis.
      next_main = main_lm_->start(true);
    }
    if (contact_kenlm_->HasUserToken(user_token_id)) {
      float ignored_contact_score = 0.0f;
      std::tie(next_contact, ignored_contact_score) =
          contact_lm_->score(in->contact_lm_state, user_token_id);
    } else {
      next_contact = contact_lm_->start(true);
    }
    if (!emitted_contact) {
      saw_base_word_before_contact = true;
    }
  }

  // Pointer identity is used by Flashlight when it merges hypotheses.  Cache
  // children by both parent state and the user token, so two dynamic labels
  // never merge merely because they reached the same underlying KenLM state.
  auto out = in->child<ContextualContactLmState>(user_token_id);
  out->main_lm_state = std::move(next_main);
  out->contact_lm_state = std::move(next_contact);
  out->saw_base_word_before_contact = saw_base_word_before_contact;
  out->emitted_contact = emitted_contact;
  out->exact_standalone_contact = exact_standalone_contact;
  return std::make_pair(std::move(out), combined_score);
}

std::pair<fl::lib::text::LMStatePtr, float> ContextualContactLm::finish(
    const fl::lib::text::LMStatePtr& state) {
  const auto in = RequireState(state);
  fl::lib::text::LMStatePtr next_main;
  fl::lib::text::LMStatePtr next_contact;
  float raw_main_score = 0.0f;
  std::tie(next_main, raw_main_score) = main_lm_->finish(in->main_lm_state);
  // Contact-pattern scores are emitted only at the verified <CONTACT>
  // transition; EOS must not add a sparse-LM probability to normal words.
  std::tie(next_contact, std::ignore) = contact_lm_->finish(in->contact_lm_state);
  auto out = in->child<ContextualContactLmState>(-1);
  out->main_lm_state = std::move(next_main);
  out->contact_lm_state = std::move(next_contact);
  out->saw_base_word_before_contact = in->saw_base_word_before_contact;
  out->emitted_contact = in->emitted_contact;
  out->exact_standalone_contact = in->exact_standalone_contact;
  return std::make_pair(std::move(out),
                        static_cast<float>(main_lm_weight_) * raw_main_score);
}

void ContextualContactLm::updateCache(
    std::vector<fl::lib::text::LMStatePtr> states) {
  std::vector<fl::lib::text::LMStatePtr> main_states;
  std::vector<fl::lib::text::LMStatePtr> contact_states;
  main_states.reserve(states.size());
  contact_states.reserve(states.size());
  std::unordered_set<const fl::lib::text::LMState*> seen_main;
  std::unordered_set<const fl::lib::text::LMState*> seen;
  for (const auto& state : states) {
    const auto typed = std::static_pointer_cast<ContextualContactLmState>(state);
    if (typed && typed->main_lm_state &&
        seen_main.insert(typed->main_lm_state.get()).second) {
      main_states.push_back(typed->main_lm_state);
    }
    if (typed && typed->contact_lm_state &&
        seen.insert(typed->contact_lm_state.get()).second) {
      contact_states.push_back(typed->contact_lm_state);
    }
  }
  if (!main_states.empty()) {
    main_lm_->updateCache(std::move(main_states));
  }
  if (!contact_states.empty()) {
    contact_lm_->updateCache(std::move(contact_states));
  }
}

}  // namespace asr_sdk::internal::flashlight_decoder
