#include "flashlight_decoder/multi_trie_lexicon_decoder.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <tuple>
#include <utility>

#include "flashlight/lib/text/decoder/Utils.h"

namespace asr_sdk::internal::flashlight_decoder {
namespace {

const fl::lib::text::TrieNode* OverlayChild(
    const fl::lib::text::TrieNode* node, int token) {
  if (!node) return nullptr;
  const auto it = node->children.find(token);
  return it == node->children.end() ? nullptr : it->second.get();
}

bool HasChildren(const CompactLexicon& base,
                 CompactLexicon::StateId base_state,
                 const fl::lib::text::TrieNode* overlay) {
  return (base_state != CompactLexicon::kInvalidState &&
          base.HasChildren(base_state)) ||
         (overlay && !overlay->children.empty());
}

bool HasLabels(const CompactLexicon& base,
               CompactLexicon::StateId base_state,
               const fl::lib::text::TrieNode* overlay) {
  return (base_state != CompactLexicon::kInvalidState &&
          !base.Labels(base_state).empty()) ||
         (overlay && !overlay->labels.empty());
}

}  // namespace

int MultiTrieLexiconDecoderState::compareNoScoreStates(
    const MultiTrieLexiconDecoderState* node) const {
  const int lm_cmp = lmState->compare(node->lmState);
  if (lm_cmp != 0) return lm_cmp > 0 ? 1 : -1;
  if (base_lex != node->base_lex) return base_lex > node->base_lex ? 1 : -1;
  if (overlay_lex != node->overlay_lex) {
    return overlay_lex > node->overlay_lex ? 1 : -1;
  }
  if (token != node->token) return token > node->token ? 1 : -1;
  if (prevBlank != node->prevBlank) {
    return prevBlank > node->prevBlank ? 1 : -1;
  }
  return 0;
}

MultiTrieLexiconDecoder::MultiTrieLexiconDecoder(
    fl::lib::text::LexiconDecoderOptions options,
    std::shared_ptr<const CompactLexicon> base_lexicon,
    fl::lib::text::TriePtr overlay_lexicon,
    fl::lib::text::LMPtr lm, int sil, int blank, int unk,
    std::vector<float> transitions, bool is_lm_token,
    bool context_lookahead)
    : options_(std::move(options)),
      base_lexicon_(std::move(base_lexicon)),
      overlay_lexicon_(std::move(overlay_lexicon)),
      lm_(std::move(lm)),
      sil_(sil),
      blank_(blank),
      unk_(unk),
      transitions_(std::move(transitions)),
      is_lm_token_(is_lm_token),
      context_lookahead_(context_lookahead) {
  if (!base_lexicon_ || !overlay_lexicon_ || !lm_) {
    throw std::invalid_argument(
        "compact multi-trie decoder requires both tries and LM");
  }
}

bool MultiTrieLexiconDecoder::IsRoot(
    const MultiTrieLexiconDecoderState& state) const {
  return state.base_lex == CompactLexicon::kRoot &&
         state.overlay_lex == overlay_lexicon_->getRoot();
}

double MultiTrieLexiconDecoder::Lookahead(
    const MultiTrieLexiconDecoderState& state) const {
  if (IsRoot(state)) return 0.0;
  double score = fl::lib::text::kNegativeInfinity;
  if (state.base_lex != CompactLexicon::kInvalidState) {
    score = base_lexicon_->Lookahead(state.base_lex, context_lookahead_);
  }
  if (state.overlay_lex) {
    score = std::max(score, static_cast<double>(state.overlay_lex->maxScore));
  }
  return score;
}

void MultiTrieLexiconDecoder::decodeBegin() {
  hyp_.clear();
  hyp_.emplace(0, std::vector<MultiTrieLexiconDecoderState>());
  hyp_[0].emplace_back(
      0.0, lm_->start(false), CompactLexicon::kRoot,
      overlay_lexicon_->getRoot(), nullptr, sil_, -1);
  decoded_frames_ = 0;
  pruned_frames_ = 0;
}

void MultiTrieLexiconDecoder::decodeStep(
    const float* emissions, int frames, int token_count) {
  const int start_frame = decoded_frames_ - pruned_frames_;
  if (hyp_.size() < static_cast<size_t>(start_frame + frames + 2)) {
    for (int i = static_cast<int>(hyp_.size());
         i < start_frame + frames + 2; ++i) {
      hyp_.emplace(i, std::vector<MultiTrieLexiconDecoderState>());
    }
  }

  std::vector<size_t> indices(token_count);
  for (int t = 0; t < frames; ++t) {
    std::iota(indices.begin(), indices.end(), 0);
    if (token_count > options_.beamSizeToken) {
      std::partial_sort(
          indices.begin(), indices.begin() + options_.beamSizeToken,
          indices.end(), [t, token_count, emissions](size_t left, size_t right) {
            return emissions[t * token_count + left] >
                   emissions[t * token_count + right];
          });
    }

    fl::lib::text::candidatesReset(
        candidates_best_score_, candidates_, candidate_ptrs_);
    for (const MultiTrieLexiconDecoderState& previous :
         hyp_[start_frame + t]) {
      const int previous_token = previous.token;
      const double previous_lookahead = Lookahead(previous);
      const bool previous_is_root = IsRoot(previous);

      for (int rank = 0;
           rank < std::min(options_.beamSizeToken, token_count); ++rank) {
        const int token = static_cast<int>(indices[rank]);
        const CompactLexicon::StateId base_child =
            base_lexicon_->Child(previous.base_lex, token);
        const fl::lib::text::TrieNode* overlay_child =
            OverlayChild(previous.overlay_lex, token);
        if (base_child == CompactLexicon::kInvalidState && !overlay_child) {
          continue;
        }

        double am_score = emissions[t * token_count + token];
        if (decoded_frames_ + t > 0 &&
            options_.criterionType == fl::lib::text::CriterionType::ASG) {
          am_score += transitions_[token * token_count + previous_token];
        }
        double score = previous.score + am_score;
        if (token == sil_) score += options_.silScore;

        fl::lib::text::LMStatePtr lm_state;
        double lm_score = 0.0;
        if (is_lm_token_) {
          std::tie(lm_state, lm_score) = lm_->score(previous.lmState, token);
        }

        if (options_.criterionType != fl::lib::text::CriterionType::CTC ||
            previous.prevBlank || token != previous_token) {
          if (HasChildren(*base_lexicon_, base_child, overlay_child)) {
            if (!is_lm_token_) {
              lm_state = previous.lmState;
              MultiTrieLexiconDecoderState child_state(
                  0, lm_state, base_child, overlay_child, nullptr, token, -1);
              const double child_lookahead = Lookahead(child_state);
              lm_score = context_lookahead_
                  ? child_lookahead - previous_lookahead
                  : static_cast<float>(child_lookahead) -
                        static_cast<float>(previous_lookahead);
            }
            fl::lib::text::candidatesAdd(
                candidates_, candidates_best_score_, options_.beamThreshold,
                score + options_.lmWeight * lm_score, lm_state, base_child,
                overlay_child, &previous, token, -1, false,
                previous.emittingModelScore + am_score,
                previous.lmScore + lm_score);
          }
        }

        auto add_word = [&](int label) {
          if (previous_is_root && previous.token == token) return;
          fl::lib::text::LMStatePtr word_lm_state = lm_state;
          double word_lm_score = lm_score;
          if (!is_lm_token_) {
            std::tie(word_lm_state, word_lm_score) =
                lm_->score(previous.lmState, label);
            word_lm_score = context_lookahead_
                ? word_lm_score - previous_lookahead
                : static_cast<float>(word_lm_score) -
                      static_cast<float>(previous_lookahead);
          }
          fl::lib::text::candidatesAdd(
              candidates_, candidates_best_score_, options_.beamThreshold,
              score + options_.lmWeight * word_lm_score +
                  options_.wordScore,
              word_lm_state, CompactLexicon::kRoot,
              overlay_lexicon_->getRoot(), &previous, token, label, false,
              previous.emittingModelScore + am_score,
              previous.lmScore + word_lm_score);
        };
        if (base_child != CompactLexicon::kInvalidState) {
          for (uint32_t label : base_lexicon_->Labels(base_child)) {
            add_word(static_cast<int>(label));
          }
        }
        auto add_overlay_words = [&](const fl::lib::text::TrieNode* node) {
          if (!node) return;
          for (int label : node->labels) add_word(label);
        };
        add_overlay_words(overlay_child);

        if (!HasLabels(*base_lexicon_, base_child, overlay_child) &&
            options_.unkScore > fl::lib::text::kNegativeInfinity) {
          if (!is_lm_token_) {
            std::tie(lm_state, lm_score) =
                lm_->score(previous.lmState, unk_);
            lm_score = context_lookahead_
                ? lm_score - previous_lookahead
                : static_cast<float>(lm_score) -
                      static_cast<float>(previous_lookahead);
          }
          fl::lib::text::candidatesAdd(
              candidates_, candidates_best_score_, options_.beamThreshold,
              score + options_.lmWeight * lm_score + options_.unkScore,
              lm_state, CompactLexicon::kRoot, overlay_lexicon_->getRoot(),
              &previous, token, unk_, false,
              previous.emittingModelScore + am_score,
              previous.lmScore + lm_score);
        }
      }

      if (options_.criterionType != fl::lib::text::CriterionType::CTC ||
          !previous.prevBlank || previous_is_root) {
        const int token = previous_is_root ? sil_ : previous_token;
        double am_score = emissions[t * token_count + token];
        if (decoded_frames_ + t > 0 &&
            options_.criterionType == fl::lib::text::CriterionType::ASG) {
          am_score += transitions_[token * token_count + previous_token];
        }
        double score = previous.score + am_score;
        if (token == sil_) score += options_.silScore;
        fl::lib::text::candidatesAdd(
            candidates_, candidates_best_score_, options_.beamThreshold,
            score, previous.lmState, previous.base_lex,
            previous.overlay_lex, &previous, token, -1, false,
            previous.emittingModelScore + am_score, previous.lmScore);
      }

      if (options_.criterionType == fl::lib::text::CriterionType::CTC) {
        const double am_score = emissions[t * token_count + blank_];
        fl::lib::text::candidatesAdd(
            candidates_, candidates_best_score_, options_.beamThreshold,
            previous.score + am_score, previous.lmState, previous.base_lex,
            previous.overlay_lex, &previous, blank_, -1, true,
            previous.emittingModelScore + am_score, previous.lmScore);
      }
    }

    fl::lib::text::candidatesStore(
        candidates_, candidate_ptrs_, hyp_[start_frame + t + 1],
        options_.beamSize, candidates_best_score_ - options_.beamThreshold,
        options_.logAdd, false);
    fl::lib::text::updateLMCache(lm_, hyp_[start_frame + t + 1]);
  }
  decoded_frames_ += frames;
}

void MultiTrieLexiconDecoder::decodeEnd() {
  fl::lib::text::candidatesReset(
      candidates_best_score_, candidates_, candidate_ptrs_);
  const auto& final_beam = hyp_[decoded_frames_ - pruned_frames_];
  const bool has_nice_ending = std::any_of(
      final_beam.begin(), final_beam.end(),
      [this](const auto& state) { return IsRoot(state); });
  for (const MultiTrieLexiconDecoderState& previous : final_beam) {
    if (!has_nice_ending || IsRoot(previous)) {
      auto finished = lm_->finish(previous.lmState);
      fl::lib::text::candidatesAdd(
          candidates_, candidates_best_score_, options_.beamThreshold,
          previous.score + options_.lmWeight * finished.second,
          finished.first, previous.base_lex, previous.overlay_lex,
          &previous, sil_, -1, false,
          previous.emittingModelScore,
          previous.lmScore + finished.second);
    }
  }
  fl::lib::text::candidatesStore(
      candidates_, candidate_ptrs_,
      hyp_[decoded_frames_ - pruned_frames_ + 1], options_.beamSize,
      candidates_best_score_ - options_.beamThreshold, options_.logAdd, true);
  ++decoded_frames_;
}

std::vector<fl::lib::text::DecodeResult>
MultiTrieLexiconDecoder::getAllFinalHypothesis() const {
  const int final_frame = decoded_frames_ - pruned_frames_;
  if (final_frame < 1) return {};
  return fl::lib::text::getAllHypothesis(
      hyp_.find(final_frame)->second, final_frame);
}

fl::lib::text::DecodeResult MultiTrieLexiconDecoder::getBestHypothesis(
    int look_back) const {
  if (decoded_frames_ - pruned_frames_ - look_back < 1) {
    return fl::lib::text::DecodeResult();
  }
  const auto* best = fl::lib::text::findBestAncestor(
      hyp_.find(decoded_frames_ - pruned_frames_)->second, look_back);
  return fl::lib::text::getHypothesis(
      best, decoded_frames_ - pruned_frames_ - look_back);
}

int MultiTrieLexiconDecoder::nDecodedFramesInBuffer() const {
  return decoded_frames_ - pruned_frames_ + 1;
}

void MultiTrieLexiconDecoder::prune(int look_back) {
  if (decoded_frames_ - pruned_frames_ - look_back < 1) return;
  const auto* best = fl::lib::text::findBestAncestor(
      hyp_.find(decoded_frames_ - pruned_frames_)->second, look_back);
  if (!best) return;
  const int start_frame = decoded_frames_ - pruned_frames_ - look_back;
  if (start_frame < 1) return;
  fl::lib::text::pruneAndNormalize(hyp_, start_frame, look_back);
  pruned_frames_ = decoded_frames_ - look_back;
}

}  // namespace asr_sdk::internal::flashlight_decoder
