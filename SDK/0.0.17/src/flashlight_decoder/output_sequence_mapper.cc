#include "flashlight_decoder/output_sequence_mapper.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <optional>
#include <queue>
#include <sstream>
#include <stdexcept>

namespace asr_sdk::internal::flashlight_decoder {
namespace {

std::string Trim(std::string text) {
  size_t begin = 0;
  while (begin < text.size() &&
         std::isspace(static_cast<unsigned char>(text[begin]))) {
    ++begin;
  }
  size_t end = text.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(text[end - 1]))) {
    --end;
  }
  return text.substr(begin, end - begin);
}

std::vector<std::string> SplitAsciiWhitespace(const std::string& text) {
  std::istringstream in(text);
  std::vector<std::string> items;
  std::string item;
  while (in >> item) {
    items.push_back(item);
  }
  return items;
}

void ParseAnchors(std::vector<std::string>* source, bool* anchor_start,
                  bool* anchor_end, const std::filesystem::path& path,
                  int line_no) {
  *anchor_start = false;
  *anchor_end = false;
  if (!source->empty() && !source->front().empty() &&
      source->front().front() == '^') {
    *anchor_start = true;
    source->front().erase(0, 1);
  }
  if (!source->empty() && !source->back().empty() &&
      source->back().back() == '$') {
    *anchor_end = true;
    source->back().pop_back();
  }
  for (const std::string& item : *source) {
    if (item.empty()) {
      throw std::runtime_error(path.string() + ":" +
                               std::to_string(line_no) +
                               ": an anchor must be attached to a source word");
    }
    if (item.find('^') != std::string::npos ||
        item.find('$') != std::string::npos) {
      throw std::runtime_error(path.string() + ":" +
                               std::to_string(line_no) +
                               ": '^' and '$' are only valid at source boundaries");
    }
  }
}

std::string KeyFor(const std::vector<int>& ids) {
  std::string key;
  for (int id : ids) {
    key += std::to_string(id);
    key.push_back('\x1f');
  }
  return key;
}

std::string TextKeyFor(const std::vector<std::string>& words) {
  std::string key;
  for (const std::string& word : words) {
    key += word;
    key.push_back('\x1f');
  }
  return key;
}

std::optional<std::vector<int>> TryResolveWords(
    const std::vector<std::string>& items, const WordDictionary& words) {
  std::vector<int> ids;
  ids.reserve(items.size());
  for (const std::string& item : items) {
    if (!words.Contains(item)) {
      return std::nullopt;
    }
    ids.push_back(words.Id(item));
  }
  return ids;
}

std::vector<int> ResolveKnownTargetIds(const std::vector<std::string>& items,
                                       const WordDictionary& words) {
  std::vector<int> ids;
  ids.reserve(items.size());
  for (const std::string& item : items) {
    ids.push_back(words.Contains(item) ? words.Id(item) : -1);
  }
  return ids;
}

}  // namespace

OutputSequenceMapper OutputSequenceMapper::Identity(
    const WordDictionary& words) {
  OutputSequenceMapper mapper;
  mapper.id_to_word_ = words.Entries();
  return mapper;
}

OutputSequenceMapper OutputSequenceMapper::Load(
    const std::filesystem::path& path, const WordDictionary& words) {
  if (path.empty()) {
    return Identity(words);
  }
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open output_mapping.txt: " +
                             path.string());
  }

  OutputSequenceMapper mapper = Identity(words);
  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    line = Trim(line);
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const std::string delimiter = " -> ";
    const size_t pos = line.find(delimiter);
    if (pos == std::string::npos ||
        line.find(delimiter, pos + delimiter.size()) != std::string::npos) {
      throw std::runtime_error(path.string() + ":" +
                               std::to_string(line_no) +
                               ": expected exact delimiter ' -> '");
    }
    std::vector<std::string> source =
        SplitAsciiWhitespace(line.substr(0, pos));
    std::vector<std::string> target =
        SplitAsciiWhitespace(line.substr(pos + delimiter.size()));
    if (source.empty() || target.empty()) {
      throw std::runtime_error(path.string() + ":" +
                               std::to_string(line_no) +
                               ": mapping source and target must be non-empty");
    }
    bool anchor_start = false;
    bool anchor_end = false;
    ParseAnchors(&source, &anchor_start, &anchor_end, path, line_no);

    std::optional<std::vector<int>> source_ids =
        TryResolveWords(source, words);
    std::optional<std::vector<int>> target_ids =
        TryResolveWords(target, words);
    if (source_ids.has_value() && target_ids.has_value()) {
      Rule rule;
      rule.line = line_no;
      rule.anchor_start = anchor_start;
      rule.anchor_end = anchor_end;
      rule.source = std::move(source_ids).value();
      rule.target = std::move(target_ids).value();
      mapper.AddRule(std::move(rule), path);
    } else {
      TextRule rule;
      rule.line = line_no;
      rule.anchor_start = anchor_start;
      rule.anchor_end = anchor_end;
      rule.source = std::move(source);
      rule.target = std::move(target);
      rule.target_ids = ResolveKnownTargetIds(rule.target, words);
      mapper.AddTextRule(std::move(rule), path);
    }
  }
  return mapper;
}

void OutputSequenceMapper::AddRule(Rule rule,
                                   const std::filesystem::path& path) {
  const std::string key = KeyFor(rule.source);
  if (source_keys_.find(key) != source_keys_.end()) {
    throw std::runtime_error(path.string() + ":" + std::to_string(rule.line) +
                             ": duplicate mapping source");
  }

  int node = 0;
  for (int id : rule.source) {
    auto it = trie_[static_cast<size_t>(node)].next.find(id);
    if (it == trie_[static_cast<size_t>(node)].next.end()) {
      const int child = static_cast<int>(trie_.size());
      trie_.push_back(TrieNode{});
      trie_[static_cast<size_t>(node)].next[id] = child;
      node = child;
    } else {
      node = it->second;
    }
  }
  trie_[static_cast<size_t>(node)].rule = static_cast<int>(rules_.size());
  max_id_source_length_ = std::max(max_id_source_length_, rule.source.size());
  source_keys_[key] = rule.line;
  rules_.push_back(std::move(rule));
}

void OutputSequenceMapper::ValidateForPreLm(
    const std::filesystem::path& path) const {
  if (!text_rules_.empty()) {
    throw std::runtime_error(
        path.string() +
        ": pre-LM mapping source and target words must all exist in words.txt");
  }
  for (const Rule& rule : rules_) {
    if (rule.anchor_end) {
      throw std::runtime_error(path.string() + ":" +
                               std::to_string(rule.line) +
                               ": '$' is not supported by streaming pre-LM mappings");
    }

    int node = 0;
    for (size_t index = 0; index < rule.source.size(); ++index) {
      if (index > 0) {
        const int shorter = trie_[static_cast<size_t>(node)].rule;
        if (shorter >= 0) {
          throw std::runtime_error(
              path.string() + ": mapping source on line " +
              std::to_string(rules_[static_cast<size_t>(shorter)].line) +
              " is a strict prefix of the source on line " +
              std::to_string(rule.line));
        }
      }
      node = trie_[static_cast<size_t>(node)].next.at(rule.source[index]);
    }

    std::queue<int> pending;
    for (const auto& item : trie_[static_cast<size_t>(node)].next) {
      pending.push(item.second);
    }
    while (!pending.empty()) {
      const int descendant = pending.front();
      pending.pop();
      const int longer = trie_[static_cast<size_t>(descendant)].rule;
      if (longer >= 0) {
        throw std::runtime_error(
            path.string() + ": mapping source on line " +
            std::to_string(rule.line) +
            " is a strict prefix of the source on line " +
            std::to_string(rules_[static_cast<size_t>(longer)].line));
      }
      for (const auto& item :
           trie_[static_cast<size_t>(descendant)].next) {
        pending.push(item.second);
      }
    }
  }
}

std::optional<OutputSequenceMapper::IdSuffixMatch>
OutputSequenceMapper::MatchIdSuffix(
    const std::vector<int>& recent_input,
    size_t absolute_input_size) const {
  int best_rule = -1;
  size_t best_length = 0;
  const size_t max_length =
      std::min(recent_input.size(), max_id_source_length_);
  for (size_t length = 1; length <= max_length; ++length) {
    const size_t begin = recent_input.size() - length;
    int node = 0;
    bool matched = true;
    for (size_t index = begin; index < recent_input.size(); ++index) {
      const auto it =
          trie_[static_cast<size_t>(node)].next.find(recent_input[index]);
      if (it == trie_[static_cast<size_t>(node)].next.end()) {
        matched = false;
        break;
      }
      node = it->second;
    }
    if (!matched) continue;
    const int rule_index = trie_[static_cast<size_t>(node)].rule;
    if (rule_index < 0) continue;
    const Rule& rule = rules_[static_cast<size_t>(rule_index)];
    const size_t absolute_begin = absolute_input_size - length;
    if (rule.anchor_start && absolute_begin != 0) continue;
    if (length > best_length) {
      best_rule = rule_index;
      best_length = length;
    }
  }
  if (best_rule < 0) return std::nullopt;
  const Rule& rule = rules_[static_cast<size_t>(best_rule)];
  return IdSuffixMatch{rule.line, rule.source.size(), &rule.target};
}

void OutputSequenceMapper::AddTextRule(
    TextRule rule, const std::filesystem::path& path) {
  const std::string key = TextKeyFor(rule.source);
  if (text_source_keys_.find(key) != text_source_keys_.end()) {
    throw std::runtime_error(path.string() + ":" + std::to_string(rule.line) +
                             ": duplicate mapping source");
  }

  int node = 0;
  for (const std::string& word : rule.source) {
    auto it = text_trie_[static_cast<size_t>(node)].next.find(word);
    if (it == text_trie_[static_cast<size_t>(node)].next.end()) {
      const int child = static_cast<int>(text_trie_.size());
      text_trie_.push_back(TextTrieNode{});
      text_trie_[static_cast<size_t>(node)].next[word] = child;
      node = child;
    } else {
      node = it->second;
    }
  }
  text_trie_[static_cast<size_t>(node)].rule =
      static_cast<int>(text_rules_.size());
  text_source_keys_[key] = rule.line;
  text_rules_.push_back(std::move(rule));
}

std::vector<int> OutputSequenceMapper::RewriteIds(
    const std::vector<int>& input) const {
  std::vector<int> output;
  size_t i = 0;
  while (i < input.size()) {
    int node = 0;
    int best_rule = -1;
    size_t best_end = i;
    size_t j = i;
    while (j < input.size()) {
      const auto it = trie_[static_cast<size_t>(node)].next.find(input[j]);
      if (it == trie_[static_cast<size_t>(node)].next.end()) {
        break;
      }
      node = it->second;
      ++j;
      const int rule = trie_[static_cast<size_t>(node)].rule;
      if (rule >= 0 &&
          (!rules_[static_cast<size_t>(rule)].anchor_start || i == 0) &&
          (!rules_[static_cast<size_t>(rule)].anchor_end || j == input.size())) {
        best_rule = rule;
        best_end = j;
      }
    }
    if (best_rule >= 0) {
      const Rule& rule = rules_[static_cast<size_t>(best_rule)];
      output.insert(output.end(), rule.target.begin(), rule.target.end());
      i = best_end;
    } else {
      output.push_back(input[i]);
      ++i;
    }
  }
  return output;
}

std::vector<DecodedWord> OutputSequenceMapper::RewriteWords(
    const std::vector<DecodedWord>& input,
    size_t absolute_input_offset) const {
  if (rules_.empty() && text_rules_.empty()) {
    return input;
  }
  std::vector<DecodedWord> output;
  size_t i = 0;
  while (i < input.size()) {
    int node = 0;
    int best_rule = -1;
    size_t best_end = i;
    size_t j = i;
    while (j < input.size()) {
      const auto it =
          trie_[static_cast<size_t>(node)].next.find(input[j].word_id);
      if (it == trie_[static_cast<size_t>(node)].next.end()) {
        break;
      }
      node = it->second;
      ++j;
      const int rule = trie_[static_cast<size_t>(node)].rule;
      if (rule >= 0 &&
          (!rules_[static_cast<size_t>(rule)].anchor_start ||
           absolute_input_offset + i == 0) &&
          (!rules_[static_cast<size_t>(rule)].anchor_end || j == input.size())) {
        best_rule = rule;
        best_end = j;
      }
    }
    int best_text_rule = -1;
    size_t best_text_end = i;
    j = i;
    node = 0;
    while (j < input.size()) {
      const auto it =
          text_trie_[static_cast<size_t>(node)].next.find(input[j].text);
      if (it == text_trie_[static_cast<size_t>(node)].next.end()) {
        break;
      }
      node = it->second;
      ++j;
      const int rule = text_trie_[static_cast<size_t>(node)].rule;
      if (rule >= 0 &&
          (!text_rules_[static_cast<size_t>(rule)].anchor_start ||
           absolute_input_offset + i == 0) &&
          (!text_rules_[static_cast<size_t>(rule)].anchor_end ||
           j == input.size())) {
        best_text_rule = rule;
        best_text_end = j;
      }
    }

    const size_t id_len = best_rule >= 0 ? best_end - i : 0;
    const size_t text_len = best_text_rule >= 0 ? best_text_end - i : 0;
    if (id_len == 0 && text_len == 0) {
      output.push_back(input[i]);
      ++i;
      continue;
    }

    if (text_len > id_len) {
      const TextRule& rule = text_rules_[static_cast<size_t>(best_text_rule)];
      const size_t source_len = best_text_end - i;
      const size_t target_len = rule.target.size();
      const int span_start = input[i].start_frame;
      const int span_end = input[best_text_end - 1].end_frame;
      for (size_t k = 0; k < target_len; ++k) {
        DecodedWord word;
        word.word_id = rule.target_ids[k];
        word.text = rule.target[k];
        if (source_len == target_len) {
          word.start_frame = input[i + k].start_frame;
          word.end_frame = input[i + k].end_frame;
          word.timestamp_derived = input[i + k].timestamp_derived;
        } else {
          const int width = span_end - span_start;
          word.start_frame =
              span_start + static_cast<int>((width * k) / target_len);
          word.end_frame =
              span_start + static_cast<int>((width * (k + 1)) / target_len);
          word.timestamp_derived = true;
        }
        output.push_back(std::move(word));
      }
      i = best_text_end;
      continue;
    }

    const Rule& rule = rules_[static_cast<size_t>(best_rule)];
    const size_t source_len = best_end - i;
    const size_t target_len = rule.target.size();
    const int span_start = input[i].start_frame;
    const int span_end = input[best_end - 1].end_frame;
    for (size_t k = 0; k < target_len; ++k) {
      DecodedWord word;
      word.word_id = rule.target[k];
      word.text = WordForId(word.word_id);
      if (source_len == target_len) {
        word.start_frame = input[i + k].start_frame;
        word.end_frame = input[i + k].end_frame;
        word.timestamp_derived = input[i + k].timestamp_derived;
      } else {
        const int width = span_end - span_start;
        word.start_frame =
            span_start + static_cast<int>((width * k) / target_len);
        word.end_frame =
            span_start + static_cast<int>((width * (k + 1)) / target_len);
        word.timestamp_derived = true;
      }
      output.push_back(std::move(word));
    }
    i = best_end;
  }
  return output;
}

std::string OutputSequenceMapper::WordForId(int id) const {
  if (id < 0 || id >= static_cast<int>(id_to_word_.size())) {
    throw std::runtime_error("mapped word id out of range: " +
                             std::to_string(id));
  }
  return id_to_word_[static_cast<size_t>(id)];
}

}  // namespace asr_sdk::internal::flashlight_decoder
