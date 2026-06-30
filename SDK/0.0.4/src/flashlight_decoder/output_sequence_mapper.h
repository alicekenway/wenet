#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_OUTPUT_SEQUENCE_MAPPER_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_OUTPUT_SEQUENCE_MAPPER_H_

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "flashlight_decoder/decoded_hypothesis.h"
#include "flashlight_decoder/word_dictionary.h"

namespace asr_sdk::internal::flashlight_decoder {

class OutputSequenceMapper {
 public:
  OutputSequenceMapper() = default;

  static OutputSequenceMapper Identity(const WordDictionary& words);
  static OutputSequenceMapper Load(const std::filesystem::path& path,
                                   const WordDictionary& words);

  int RuleCount() const { return static_cast<int>(rules_.size()); }
  bool empty() const { return rules_.empty(); }
  std::vector<int> RewriteIds(const std::vector<int>& input) const;
  std::vector<DecodedWord> RewriteWords(
      const std::vector<DecodedWord>& input) const;

 private:
  struct Rule {
    int line = 0;
    std::vector<int> source;
    std::vector<int> target;
  };

  struct TrieNode {
    std::unordered_map<int, int> next;
    int rule = -1;
  };

  std::string WordForId(int id) const;
  void AddRule(Rule rule, const std::filesystem::path& path);

  std::vector<std::string> id_to_word_;
  std::vector<Rule> rules_;
  std::vector<TrieNode> trie_{{}};
  std::unordered_map<std::string, int> source_keys_;
};

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_OUTPUT_SEQUENCE_MAPPER_H_
