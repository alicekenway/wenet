#include "decoder/ctc_prefix_decoder.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace wenet_sdk::internal {

namespace {

constexpr float kNegInf = -std::numeric_limits<float>::max();

float LogAdd(float x, float y) {
  if (x <= kNegInf / 2) {
    return y;
  }
  if (y <= kNegInf / 2) {
    return x;
  }
  const float max_value = std::max(x, y);
  return max_value + std::log(std::exp(x - max_value) +
                              std::exp(y - max_value));
}

std::vector<int> TopKIndexes(const std::vector<float>& values, int k) {
  std::vector<int> indexes(values.size());
  for (size_t i = 0; i < indexes.size(); ++i) {
    indexes[i] = static_cast<int>(i);
  }
  k = std::max(0, std::min(k, static_cast<int>(indexes.size())));
  if (k < static_cast<int>(indexes.size())) {
    std::partial_sort(indexes.begin(), indexes.begin() + k, indexes.end(),
                      [&values](int a, int b) { return values[a] > values[b]; });
    indexes.resize(static_cast<size_t>(k));
  } else {
    std::sort(indexes.begin(), indexes.end(),
              [&values](int a, int b) { return values[a] > values[b]; });
  }
  return indexes;
}

}  // namespace

float CtcPrefixDecoder::PrefixScore::Score() const {
  return LogAdd(blank, non_blank);
}

float CtcPrefixDecoder::PrefixScore::ViterbiScore() const {
  return std::max(viterbi_blank, viterbi_non_blank);
}

const std::vector<int>& CtcPrefixDecoder::PrefixScore::Times() const {
  return viterbi_blank > viterbi_non_blank ? times_blank : times_non_blank;
}

CtcPrefixDecoder::CtcPrefixDecoder(CtcPrefixDecoderOptions options)
    : options_(options) {
  Reset();
}

void CtcPrefixDecoder::Reset() {
  abs_time_step_ = 0;
  cur_hyps_.clear();
  hypotheses_.clear();
  likelihood_.clear();
  times_.clear();

  PrefixScore init;
  init.blank = 0.0f;
  init.non_blank = kNegInf;
  init.viterbi_blank = 0.0f;
  init.viterbi_non_blank = kNegInf;

  std::vector<int> empty;
  cur_hyps_[empty] = init;
  hypotheses_.push_back(empty);
  likelihood_.push_back(init.Score());
  times_.push_back(empty);
}

void CtcPrefixDecoder::Advance(
    const std::vector<std::vector<float>>& log_probs) {
  if (log_probs.empty()) {
    return;
  }

  for (const auto& frame : log_probs) {
    if (frame.empty()) {
      ++abs_time_step_;
      continue;
    }
    const int first_beam_size =
        std::min(static_cast<int>(frame.size()), options_.first_beam_size);
    const auto topk = TopKIndexes(frame, first_beam_size);
    std::unordered_map<std::vector<int>, PrefixScore, PrefixHash> next_hyps;

    for (int id : topk) {
      const float prob = frame[static_cast<size_t>(id)];
      for (const auto& item : cur_hyps_) {
        const auto& prefix = item.first;
        const auto& prefix_score = item.second;
        if (id == options_.blank_id) {
          PrefixScore& next = next_hyps[prefix];
          next.blank = LogAdd(next.blank, prefix_score.Score() + prob);
          const float viterbi = prefix_score.ViterbiScore() + prob;
          if (next.viterbi_blank < viterbi) {
            next.viterbi_blank = viterbi;
            next.times_blank = prefix_score.Times();
          }
        } else if (!prefix.empty() && id == prefix.back()) {
          PrefixScore& same = next_hyps[prefix];
          same.non_blank =
              LogAdd(same.non_blank, prefix_score.non_blank + prob);
          const float same_viterbi = prefix_score.viterbi_non_blank + prob;
          if (same.viterbi_non_blank < same_viterbi) {
            same.viterbi_non_blank = same_viterbi;
            same.times_non_blank = prefix_score.times_non_blank;
            if (!same.times_non_blank.empty() &&
                same.current_token_logp < prob) {
              same.current_token_logp = prob;
              same.times_non_blank.back() = abs_time_step_;
            }
          }

          std::vector<int> extended(prefix);
          extended.push_back(id);
          PrefixScore& next = next_hyps[extended];
          next.non_blank =
              LogAdd(next.non_blank, prefix_score.blank + prob);
          const float next_viterbi = prefix_score.viterbi_blank + prob;
          if (next.viterbi_non_blank < next_viterbi) {
            next.viterbi_non_blank = next_viterbi;
            next.current_token_logp = prob;
            next.times_non_blank = prefix_score.times_blank;
            next.times_non_blank.push_back(abs_time_step_);
          }
        } else {
          std::vector<int> extended(prefix);
          extended.push_back(id);
          PrefixScore& next = next_hyps[extended];
          next.non_blank =
              LogAdd(next.non_blank, prefix_score.Score() + prob);
          const float next_viterbi = prefix_score.ViterbiScore() + prob;
          if (next.viterbi_non_blank < next_viterbi) {
            next.viterbi_non_blank = next_viterbi;
            next.current_token_logp = prob;
            next.times_non_blank = prefix_score.Times();
            next.times_non_blank.push_back(abs_time_step_);
          }
        }
      }
    }

    std::vector<std::pair<std::vector<int>, PrefixScore>> arr(next_hyps.begin(),
                                                              next_hyps.end());
    const auto compare = [](const auto& a, const auto& b) {
      return a.second.Score() > b.second.Score();
    };
    const int second_beam_size =
        std::min(static_cast<int>(arr.size()), options_.second_beam_size);
    if (second_beam_size < static_cast<int>(arr.size())) {
      std::nth_element(arr.begin(), arr.begin() + second_beam_size, arr.end(),
                       compare);
      arr.resize(static_cast<size_t>(second_beam_size));
    }
    std::sort(arr.begin(), arr.end(), compare);
    UpdateHypotheses(arr);
    ++abs_time_step_;
  }
}

DecodeResult CtcPrefixDecoder::PartialResult() const {
  return BuildResult();
}

DecodeResult CtcPrefixDecoder::Finalize() { return BuildResult(); }

DecodeResult CtcPrefixDecoder::BuildResult() const {
  DecodeResult result;
  if (hypotheses_.empty()) {
    return result;
  }
  result.token_ids = hypotheses_.front();
  if (!times_.empty()) {
    result.frame_indexes = times_.front();
  }
  if (!likelihood_.empty()) {
    result.confidence = std::exp(likelihood_.front() /
                                 std::max<size_t>(1, result.token_ids.size()));
  }
  if (!result.token_ids.empty()) {
    NBestPath best;
    best.token_ids = result.token_ids;
    best.frame_indexes = result.frame_indexes;
    best.total_score = likelihood_.empty() ? 0.0f : likelihood_.front();
    result.nbest.push_back(std::move(best));
  }
  return result;
}

void CtcPrefixDecoder::UpdateHypotheses(
    const std::vector<std::pair<std::vector<int>, PrefixScore>>& hyps) {
  cur_hyps_.clear();
  hypotheses_.clear();
  likelihood_.clear();
  times_.clear();
  for (const auto& item : hyps) {
    cur_hyps_[item.first] = item.second;
    hypotheses_.push_back(item.first);
    likelihood_.push_back(item.second.Score());
    times_.push_back(item.second.Times());
  }
}

}  // namespace wenet_sdk::internal
