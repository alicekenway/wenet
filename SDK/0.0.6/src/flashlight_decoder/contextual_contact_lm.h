#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_CONTEXTUAL_CONTACT_LM_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_CONTEXTUAL_CONTACT_LM_H_

#include <memory>
#include <vector>

#include "flashlight/lib/text/decoder/lm/LM.h"
#include "flashlight/lib/text/decoder/lm/KenLM.h"
#include "flashlight_decoder/dynamic_contact_lexicon.h"

namespace asr_sdk::internal::flashlight_decoder {

struct ContextualContactLmState : fl::lib::text::LMState {
  fl::lib::text::LMStatePtr main_lm_state;
  fl::lib::text::LMStatePtr contact_lm_state;
  bool saw_base_word_before_contact = false;
  bool emitted_contact = false;
  bool exact_standalone_contact = false;
};

// Composite LM for the contact decoder.  Ordinary words use the main LM;
// the sparse contact LM is advanced only to retain pattern history.  A dynamic
// contact consumes one <CONTACT> transition and receives the configured
// pattern bonus when that transition matches a terminal pattern rule.
class ContextualContactLm final : public fl::lib::text::LM {
 public:
  ContextualContactLm(fl::lib::text::LMPtr main_lm,
                      fl::lib::text::LMPtr contact_lm,
                      std::shared_ptr<const DynamicContactLexicon> contacts,
                      int contact_class_word_id, double main_lm_weight,
                      double contact_lm_weight,
                      double accumulation_factor, float word_score);

  fl::lib::text::LMStatePtr start(bool start_with_nothing) override;
  std::pair<fl::lib::text::LMStatePtr, float> score(
      const fl::lib::text::LMStatePtr& state, int user_token_id) override;
  std::pair<fl::lib::text::LMStatePtr, float> finish(
      const fl::lib::text::LMStatePtr& state) override;
  void updateCache(std::vector<fl::lib::text::LMStatePtr> states) override;

 private:
  std::shared_ptr<ContextualContactLmState> RequireState(
      const fl::lib::text::LMStatePtr& state) const;
  static float NegativeInfinity();

  fl::lib::text::LMPtr main_lm_;
  fl::lib::text::LMPtr contact_lm_;
  std::shared_ptr<fl::lib::text::KenLM> main_kenlm_;
  std::shared_ptr<fl::lib::text::KenLM> contact_kenlm_;
  std::shared_ptr<const DynamicContactLexicon> contacts_;
  int contact_class_word_id_ = -1;
  double main_lm_weight_ = 1.0;
  double contact_lm_weight_ = 1.0;
  double accumulation_factor_ = 0.5;
  float word_score_ = 0.0f;
};

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_CONTEXTUAL_CONTACT_LM_H_
