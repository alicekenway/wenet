#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_OUTPUT_SEQUENCE_MAPPER_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_OUTPUT_SEQUENCE_MAPPER_H_

#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "flashlight_decoder/decoded_hypothesis.h"
#include "flashlight_decoder/word_dictionary.h"

namespace asr_sdk::internal::flashlight_decoder {

class OutputSequenceMapper {
 public:
  struct IdSuffixMatch {
    int line = 0;
    size_t source_length = 0;
    const std::vector<int>* target = nullptr;
  };

  OutputSequenceMapper() = default;

  static OutputSequenceMapper Identity(const WordDictionary& words);
  static OutputSequenceMapper Load(const std::filesystem::path& path,
                                   const WordDictionary& words);

  int RuleCount() const {
    return static_cast<int>(rules_.size() + text_rules_.size());
  }
  bool empty() const { return rules_.empty() && text_rules_.empty(); }
  size_t MaxIdSourceLength() const { return max_id_source_length_; }
  void ValidateForPreLm(const std::filesystem::path& path) const;
  std::optional<IdSuffixMatch> MatchIdSuffix(
      const std::vector<int>& recent_input,
      size_t absolute_input_size) const;
  std::vector<int> RewriteIds(const std::vector<int>& input) const;
  std::vector<DecodedWord> RewriteWords(
      const std::vector<DecodedWord>& input,
      size_t absolute_input_offset = 0) const;

 private:
  struct Rule {
    int line = 0;
    bool anchor_start = false;
    bool anchor_end = false;
    std::vector<int> source;
    std::vector<int> target;
  };

  struct TextRule {
    int line = 0;
    bool anchor_start = false;
    bool anchor_end = false;
    std::vector<std::string> source;
    std::vector<std::string> target;
    std::vector<int> target_ids;
  };

  struct TrieNode {
    std::unordered_map<int, int> next;
    int rule = -1;
  };

  struct TextTrieNode {
    std::unordered_map<std::string, int> next;
    int rule = -1;
  };

  std::string WordForId(int id) const;
  void AddRule(Rule rule, const std::filesystem::path& path);
  void AddTextRule(TextRule rule, const std::filesystem::path& path);

  std::vector<std::string> id_to_word_;
  std::vector<Rule> rules_;
  std::vector<TextRule> text_rules_;
  std::vector<TrieNode> trie_{{}};
  std::vector<TextTrieNode> text_trie_{{}};
  std::unordered_map<std::string, int> source_keys_;
  std::unordered_map<std::string, int> text_source_keys_;
  size_t max_id_source_length_ = 0;
};

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_OUTPUT_SEQUENCE_MAPPER_H_
