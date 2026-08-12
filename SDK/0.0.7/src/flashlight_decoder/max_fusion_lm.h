#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_MAX_FUSION_LM_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_MAX_FUSION_LM_H_

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "flashlight/lib/text/decoder/lm/KenLM.h"
#include "flashlight/lib/text/decoder/lm/LM.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"
#include "package/model_package.h"

namespace asr_sdk::internal::flashlight_decoder {

class DynamicContactLexicon;

struct MaxFusionLmState : fl::lib::text::LMState {
  std::vector<fl::lib::text::LMStatePtr> model_states;
  std::string winner_filename;
  std::string winner_type;
  float raw_score = 0.0f;
  float reference_score = 0.0f;
  float adjusted_score = 0.0f;
  float weighted_score = 0.0f;
};

struct LoadedFixedLm {
  FixedLmConfig config;
  std::shared_ptr<fl::lib::text::KenLM> lm;
  std::unordered_set<int> slot_word_ids;
};

// Scores one fixed LM per completed transition.  Normal words select the
// maximum weighted relative n-gram score; dynamic slots select the maximum
// matching pattern-bias score.  Every model state still observes every event.
class MaxFusionLm final : public fl::lib::text::LM {
 public:
  MaxFusionLm(std::vector<FixedLmConfig> configs,
              const fl::lib::text::Dictionary& words,
              std::shared_ptr<const DynamicContactLexicon> dynamic_slots,
              float word_score);

  fl::lib::text::LMStatePtr start(bool start_with_nothing) override;
  std::pair<fl::lib::text::LMStatePtr, float> score(
      const fl::lib::text::LMStatePtr& state, int user_token_id) override;
  std::pair<fl::lib::text::LMStatePtr, float> finish(
      const fl::lib::text::LMStatePtr& state) override;
  void updateCache(std::vector<fl::lib::text::LMStatePtr> states) override;

  const std::vector<LoadedFixedLm>& Models() const { return models_; }

 private:
  std::shared_ptr<MaxFusionLmState> RequireState(
      const fl::lib::text::LMStatePtr& state) const;
  float NgramContribution(const LoadedFixedLm& model,
                          const fl::lib::text::LMStatePtr& state,
                          int word_id,
                          fl::lib::text::LMStatePtr* next,
                          float* raw, float* reference,
                          float* adjusted) const;

  std::vector<LoadedFixedLm> models_;
  std::shared_ptr<const DynamicContactLexicon> dynamic_slots_;
  float word_score_ = 0.0f;
};

}  // namespace asr_sdk::internal::flashlight_decoder

#endif
