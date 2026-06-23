#include "decoder/ctc_wfst_decoder.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

#include "decoder/greedy_ctc_decoder.h"

#ifdef WENETSDK_ENABLE_OPENFST
#include "fst/fstlib.h"  // NOLINT
#endif

namespace wenet_sdk::internal {
namespace {

struct CollapsedPath {
  std::vector<int> token_ids;
  std::vector<int> frame_indexes;
};

#ifdef WENETSDK_ENABLE_OPENFST
CollapsedPath CollapseTokenFrames(const std::vector<int>& token_ids,
                                  const std::vector<int>& frame_indexes,
                                  int blank_id) {
  CollapsedPath collapsed;
  int prev = -1;
  for (size_t i = 0; i < token_ids.size(); ++i) {
    const int id = token_ids[i];
    if (id == blank_id) {
      prev = id;
      continue;
    }
    if (id != prev) {
      collapsed.token_ids.push_back(id);
      if (i < frame_indexes.size()) {
        collapsed.frame_indexes.push_back(frame_indexes[i]);
      }
    }
    prev = id;
  }
  return collapsed;
}

using StateId = fst::StdArc::StateId;

struct WfstToken {
  float cost = std::numeric_limits<float>::infinity();
  std::vector<int> token_ids;
  std::vector<int> word_ids;
  std::vector<int> frame_indexes;
};

bool BetterCost(float candidate, float current) {
  return candidate + 1.0e-6f < current;
}

#endif

}  // namespace

struct CtcWfstDecoder::Impl {
#ifdef WENETSDK_ENABLE_OPENFST
  std::unique_ptr<fst::StdVectorFst> graph;
  std::unordered_map<StateId, WfstToken> active;
  CtcWfstDecoderOptions options;
  int frame = 0;

  explicit Impl(CtcWfstDecoderOptions opts) : options(opts) {}

  Status Load(const std::filesystem::path& path) {
    graph.reset(fst::StdVectorFst::Read(path.string()));
    if (!graph) {
      return Status::InvalidArgument("failed to read OpenFst graph: " +
                                     path.string());
    }
    if (graph->Start() == fst::kNoStateId) {
      return Status::InvalidArgument("OpenFst graph has no start state: " +
                                     path.string());
    }
    Reset();
    return Status::OK();
  }

  void Reset() {
    frame = 0;
    active.clear();
    WfstToken token;
    token.cost = 0.0f;
    active[graph->Start()] = std::move(token);
    EpsilonClosure(&active);
  }

  void Advance(const std::vector<std::vector<float>>& log_probs) {
    if (!graph || log_probs.empty()) {
      return;
    }
    for (const auto& logp : log_probs) {
      EpsilonClosure(&active);
      std::unordered_map<StateId, WfstToken> next;
      for (const auto& item : active) {
        const StateId state = item.first;
        const WfstToken& token = item.second;
        for (fst::ArcIterator<fst::StdVectorFst> aiter(*graph, state);
             !aiter.Done(); aiter.Next()) {
          const auto& arc = aiter.Value();
          if (arc.ilabel == 0) {
            continue;
          }
          const int token_id = static_cast<int>(arc.ilabel) - 1;
          if (token_id < 0 || token_id >= static_cast<int>(logp.size())) {
            continue;
          }
          WfstToken candidate = token;
          candidate.cost +=
              -logp[static_cast<size_t>(token_id)] * options.acoustic_scale +
              arc.weight.Value() * options.lm_scale + options.length_penalty;
          candidate.token_ids.push_back(token_id);
          candidate.frame_indexes.push_back(frame);
          if (arc.olabel > 0) {
            candidate.word_ids.push_back(static_cast<int>(arc.olabel));
          }
          PutIfBetter(&next, arc.nextstate, std::move(candidate));
        }
      }
      if (!next.empty()) {
        EpsilonClosure(&next);
        Prune(&next);
        active = std::move(next);
      }
      ++frame;
    }
  }

  DecodeResult Result(bool require_final) const {
    if (!graph || active.empty()) {
      return DecodeResult{};
    }

    auto closed = active;
    const_cast<Impl*>(this)->EpsilonClosure(&closed);
    StateId best_state = fst::kNoStateId;
    WfstToken best;
    best.cost = std::numeric_limits<float>::infinity();

    for (auto& item : closed) {
      float cost = item.second.cost;
      const auto final_weight = graph->Final(item.first);
      const bool is_final = std::isfinite(final_weight.Value());
      if (require_final && !is_final) {
        continue;
      }
      if (is_final) {
        cost += final_weight.Value() * options.lm_scale;
      }
      if (BetterCost(cost, best.cost)) {
        best_state = item.first;
        best = item.second;
        best.cost = cost;
      }
    }

    if (best_state == fst::kNoStateId && require_final) {
      for (const auto& item : closed) {
        if (BetterCost(item.second.cost, best.cost)) {
          best_state = item.first;
          best = item.second;
        }
      }
    }

    DecodeResult result;
    if (best_state == fst::kNoStateId) {
      return result;
    }
    result.word_ids = best.word_ids;
    const auto collapsed =
        CollapseTokenFrames(best.token_ids, best.frame_indexes, options.blank_id);
    result.token_ids = collapsed.token_ids;
    result.frame_indexes = collapsed.frame_indexes;
    const int length =
        std::max<int>(1, static_cast<int>(result.word_ids.empty()
                                              ? result.token_ids.size()
                                              : result.word_ids.size()));
    result.confidence = std::exp(-best.cost / length);
    NBestPath path;
    path.token_ids = result.token_ids;
    path.word_ids = result.word_ids;
    path.frame_indexes = result.frame_indexes;
    path.total_score = -best.cost;
    result.nbest.push_back(std::move(path));
    return result;
  }

  void EpsilonClosure(std::unordered_map<StateId, WfstToken>* tokens) {
    using QueueItem = std::pair<float, StateId>;
    std::priority_queue<QueueItem, std::vector<QueueItem>,
                        std::greater<QueueItem>>
        queue;
    for (const auto& item : *tokens) {
      queue.emplace(item.second.cost, item.first);
    }

    while (!queue.empty()) {
      const auto [cost, state] = queue.top();
      queue.pop();
      const auto current_it = tokens->find(state);
      if (current_it == tokens->end() || cost > current_it->second.cost + 1e-6f) {
        continue;
      }
      for (fst::ArcIterator<fst::StdVectorFst> aiter(*graph, state);
           !aiter.Done(); aiter.Next()) {
        const auto& arc = aiter.Value();
        if (arc.ilabel != 0) {
          continue;
        }
        WfstToken candidate = current_it->second;
        candidate.cost += arc.weight.Value() * options.lm_scale;
        if (arc.olabel > 0) {
          candidate.word_ids.push_back(static_cast<int>(arc.olabel));
        }
        const float candidate_cost = candidate.cost;
        if (PutIfBetter(tokens, arc.nextstate, std::move(candidate))) {
          queue.emplace(candidate_cost, arc.nextstate);
        }
      }
    }
  }

  bool PutIfBetter(std::unordered_map<StateId, WfstToken>* tokens,
                   StateId state, WfstToken candidate) const {
    auto it = tokens->find(state);
    if (it == tokens->end() || BetterCost(candidate.cost, it->second.cost)) {
      (*tokens)[state] = std::move(candidate);
      return true;
    }
    return false;
  }

  void Prune(std::unordered_map<StateId, WfstToken>* tokens) const {
    if (tokens->empty()) {
      return;
    }
    float best = std::numeric_limits<float>::infinity();
    for (const auto& item : *tokens) {
      best = std::min(best, item.second.cost);
    }

    for (auto it = tokens->begin(); it != tokens->end();) {
      if (it->second.cost > best + options.beam) {
        it = tokens->erase(it);
      } else {
        ++it;
      }
    }

    if (options.max_active > 0 &&
        static_cast<int>(tokens->size()) > options.max_active) {
      std::vector<std::pair<StateId, WfstToken>> sorted(tokens->begin(),
                                                        tokens->end());
      std::nth_element(
          sorted.begin(), sorted.begin() + options.max_active, sorted.end(),
          [](const auto& a, const auto& b) {
            return a.second.cost < b.second.cost;
          });
      sorted.resize(static_cast<size_t>(options.max_active));
      tokens->clear();
      for (auto& item : sorted) {
        (*tokens)[item.first] = std::move(item.second);
      }
    }
  }
#endif
};

Status CtcWfstDecoder::ValidateGraphPath(const std::filesystem::path& path) {
  if (!std::filesystem::exists(path)) {
    return Status::NotFound("WFST graph is missing: " + path.string());
  }
  if (std::filesystem::is_empty(path)) {
    return Status::InvalidArgument("WFST graph is empty: " + path.string());
  }
#ifdef WENETSDK_ENABLE_OPENFST
  std::unique_ptr<fst::StdVectorFst> graph(fst::StdVectorFst::Read(path.string()));
  if (!graph) {
    return Status::InvalidArgument("failed to read OpenFst graph: " +
                                   path.string());
  }
  if (graph->Start() == fst::kNoStateId) {
    return Status::InvalidArgument("OpenFst graph has no start state: " +
                                   path.string());
  }
#endif
  return Status::OK();
}

CtcWfstDecoder::CtcWfstDecoder(CtcWfstDecoderOptions options,
                               std::filesystem::path graph_path)
    : fallback_(CtcPrefixDecoderOptions{options.blank_id, 10, 10}),
      graph_path_(std::move(graph_path)),
      options_(options) {
#ifdef WENETSDK_ENABLE_OPENFST
  impl_ = std::make_unique<Impl>(options_);
  auto status = impl_->Load(graph_path_);
  if (!status.ok()) {
    impl_.reset();
  }
#endif
}

CtcWfstDecoder::~CtcWfstDecoder() = default;

void CtcWfstDecoder::Reset() {
  fallback_.Reset();
#ifdef WENETSDK_ENABLE_OPENFST
  if (impl_) {
    impl_->Reset();
  }
#endif
}

void CtcWfstDecoder::Advance(
    const std::vector<std::vector<float>>& log_probs) {
#ifdef WENETSDK_ENABLE_OPENFST
  if (impl_) {
    impl_->Advance(log_probs);
    return;
  }
#endif
  fallback_.Advance(log_probs);
}

DecodeResult CtcWfstDecoder::PartialResult() const {
#ifdef WENETSDK_ENABLE_OPENFST
  if (impl_) {
    return impl_->Result(false);
  }
#endif
  return fallback_.PartialResult();
}

DecodeResult CtcWfstDecoder::Finalize() {
#ifdef WENETSDK_ENABLE_OPENFST
  if (impl_) {
    return impl_->Result(true);
  }
#endif
  return fallback_.Finalize();
}

}  // namespace wenet_sdk::internal
