#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_MULTI_TRIE_LEXICON_DECODER_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_MULTI_TRIE_LEXICON_DECODER_H_

#include <unordered_map>
#include <vector>

#include "flashlight/lib/text/decoder/Decoder.h"
#include "flashlight/lib/text/decoder/LexiconDecoder.h"
#include "flashlight/lib/text/decoder/Trie.h"
#include "flashlight/lib/text/decoder/lm/LM.h"
#include "flashlight_decoder/flashlight_decoder_resource.h"

namespace asr_sdk::internal::flashlight_decoder {

// Decoder state for a logical union of one immutable base trie and one small
// runtime overlay trie. A path may be present in either trie or in both.
struct MultiTrieLexiconDecoderState {
  double score;
  fl::lib::text::LMStatePtr lmState;
  const fl::lib::text::TrieNode* base_lex;
  const fl::lib::text::TrieNode* overlay_lex;
  const MultiTrieLexiconDecoderState* parent;
  int token;
  int word;
  bool prevBlank;
  double emittingModelScore;
  double lmScore;

  MultiTrieLexiconDecoderState(
      double score, const fl::lib::text::LMStatePtr& lm_state,
      const fl::lib::text::TrieNode* base_lex,
      const fl::lib::text::TrieNode* overlay_lex,
      const MultiTrieLexiconDecoderState* parent, int token, int word,
      bool prev_blank = false, double emitting_model_score = 0,
      double lm_score = 0)
      : score(score),
        lmState(lm_state),
        base_lex(base_lex),
        overlay_lex(overlay_lex),
        parent(parent),
        token(token),
        word(word),
        prevBlank(prev_blank),
        emittingModelScore(emitting_model_score),
        lmScore(lm_score) {}

  int compareNoScoreStates(const MultiTrieLexiconDecoderState* node) const;
  int getWord() const { return word; }
  bool isComplete() const { return !parent || parent->word >= 0; }
};

class MultiTrieLexiconDecoder final : public fl::lib::text::Decoder {
 public:
  MultiTrieLexiconDecoder(
      fl::lib::text::LexiconDecoderOptions options,
      fl::lib::text::TriePtr base_lexicon,
      fl::lib::text::TriePtr overlay_lexicon,
      std::shared_ptr<const RuntimeTrieLookahead> runtime_lookahead,
      fl::lib::text::LMPtr lm, int sil, int blank, int unk,
      std::vector<float> transitions, bool is_lm_token);

  void decodeBegin() override;
  void decodeStep(const float* emissions, int frames, int token_count) override;
  void decodeEnd() override;
  void prune(int look_back = 0) override;
  int nDecodedFramesInBuffer() const override;
  fl::lib::text::DecodeResult getBestHypothesis(
      int look_back = 0) const override;
  std::vector<fl::lib::text::DecodeResult> getAllFinalHypothesis()
      const override;

 private:
  bool IsRoot(const MultiTrieLexiconDecoderState& state) const;
  double Lookahead(const MultiTrieLexiconDecoderState& state) const;

  fl::lib::text::LexiconDecoderOptions options_;
  fl::lib::text::TriePtr base_lexicon_;
  fl::lib::text::TriePtr overlay_lexicon_;
  std::shared_ptr<const RuntimeTrieLookahead> runtime_lookahead_;
  fl::lib::text::LMPtr lm_;
  int sil_;
  int blank_;
  int unk_;
  std::vector<float> transitions_;
  bool is_lm_token_;
  std::vector<MultiTrieLexiconDecoderState> candidates_;
  std::vector<MultiTrieLexiconDecoderState*> candidate_ptrs_;
  double candidates_best_score_;
  std::unordered_map<int, std::vector<MultiTrieLexiconDecoderState>> hyp_;
  int decoded_frames_ = 0;
  int pruned_frames_ = 0;
};

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_MULTI_TRIE_LEXICON_DECODER_H_
