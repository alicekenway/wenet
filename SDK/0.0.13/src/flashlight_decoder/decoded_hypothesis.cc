#include "flashlight_decoder/decoded_hypothesis.h"

#include <algorithm>
#include <unordered_map>

namespace asr_sdk::internal::flashlight_decoder {
namespace {

std::string KeyForWords(const std::vector<DecodedWord>& words) {
  std::string key;
  for (const DecodedWord& word : words) {
    key += std::to_string(word.word_id);
    key.push_back('\x1f');
  }
  return key;
}

std::string SemanticKeyForHypothesis(const DecodedHypothesis& hyp) {
  std::string key;
  for (const DecodedWord& word : hyp.mapped_words) {
    key += word.text;
    key.push_back('\x1f');
  }
  key.push_back('\x1e');
  for (const DecodedEntity& entity : hyp.entities) {
    key += entity.type;
    key.push_back('\x1f');
    key += entity.slot_token;
    key.push_back('\x1f');
    key += entity.text;
    key.push_back('\x1f');
    std::vector<std::string> candidate_ids;
    candidate_ids.reserve(entity.candidates.size());
    for (const DecodedContactCandidate& candidate : entity.candidates) {
      candidate_ids.push_back(candidate.value_id);
    }
    std::sort(candidate_ids.begin(), candidate_ids.end());
    for (const std::string& candidate_id : candidate_ids) {
      key += candidate_id;
      key.push_back('\x1f');
    }
    key.push_back('\x1e');
  }
  return key;
}

}  // namespace

std::vector<int> WordIds(const std::vector<DecodedWord>& words) {
  std::vector<int> ids;
  ids.reserve(words.size());
  for (const DecodedWord& word : words) {
    ids.push_back(word.word_id);
  }
  return ids;
}

std::string JoinWords(const std::vector<DecodedWord>& words,
                      const std::string& separator) {
  std::string text;
  for (size_t i = 0; i < words.size(); ++i) {
    if (i != 0) {
      text += separator;
    }
    text += words[i].text;
  }
  return text;
}

std::vector<DecodedHypothesis> DeduplicateDecodedHypotheses(
    std::vector<DecodedHypothesis> hyps, int limit, bool contact_aware) {
  std::vector<DecodedHypothesis> out;
  std::unordered_map<std::string, size_t> best_by_key;
  for (DecodedHypothesis& hyp : hyps) {
    const std::string key = contact_aware ? SemanticKeyForHypothesis(hyp)
                                          : KeyForWords(hyp.mapped_words);
    const auto it = best_by_key.find(key);
    if (it == best_by_key.end()) {
      best_by_key.emplace(key, out.size());
      out.push_back(std::move(hyp));
      continue;
    }
    DecodedHypothesis& existing = out[it->second];
    if (hyp.total_score > existing.total_score) {
      existing = std::move(hyp);
    }
  }
  std::sort(out.begin(), out.end(), [](const DecodedHypothesis& left,
                                       const DecodedHypothesis& right) {
    return left.total_score > right.total_score;
  });
  if (limit > 0 && static_cast<int>(out.size()) > limit) {
    out.resize(static_cast<size_t>(limit));
  }
  return out;
}

}  // namespace asr_sdk::internal::flashlight_decoder
