/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "flashlight/lib/text/Defines.h"
#include "flashlight/lib/text/decoder/lm/LM.h"
#include "flashlight/lib/text/dictionary/Dictionary.h"

// Forward declarations to avoid including KenLM headers
namespace lm {
namespace base {

class Vocabulary;
class Model;

} // namespace base
namespace ngram {

class State;

} // namespace ngram
} // namespace lm

namespace fl {
namespace lib {
namespace text {

/**
 * KenLMState is a state object from KenLM, which  contains context length,
 * indicies and compare functions
 * https://github.com/kpu/kenlm/blob/master/lm/state.hh.
 */
struct FL_TEXT_API KenLMState : LMState {
  KenLMState();
  ~KenLMState();
  std::unique_ptr<lm::ngram::State> ken_;
  lm::ngram::State* ken() {
    return ken_.get();
  }
};

/**
 * KenLM extends LM by using the toolkit https://kheafield.com/code/kenlm/.
 */
class FL_TEXT_API KenLM : public LM {
 public:
  struct ScoreWithMetadata {
    LMStatePtr state;
    float score = 0.0f;
    int ngram_length = 0;
  };

  KenLM(const std::string& path, const Dictionary& usrTknDict);

  LMStatePtr start(bool startWithNothing) override;

  std::pair<LMStatePtr, float> score(
      const LMStatePtr& state,
      const int usrTokenIdx) override;

  // Like score(), but also reports how many words of the stored n-gram were
  // used.  Pattern-bias contact LMs use this to distinguish a real terminal
  // rule from a backed-off <CONTACT> unigram.
  ScoreWithMetadata scoreWithMetadata(const LMStatePtr& state,
                                      int usrTokenIdx);

  std::pair<LMStatePtr, float> finish(const LMStatePtr& state) override;

  // True only when `word` is explicitly present in the KenLM vocabulary; it
  // distinguishes a real class word from the model's <unk> fallback.
  bool HasWord(const std::string& word) const;

  // True only when this words.txt ID maps to an explicit KenLM vocabulary
  // item, rather than KenLM's implicit <unk> fallback.
  bool HasUserToken(int usr_token_idx) const;

 private:
  std::shared_ptr<lm::base::Model> model_;
  const lm::base::Vocabulary* vocab_;
  std::vector<bool> usrTokenPresent_;
};

using KenLMPtr = std::shared_ptr<KenLM>;

} // namespace text
} // namespace lib
} // namespace fl
