#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_WORD_DICTIONARY_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_WORD_DICTIONARY_H_

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

namespace asr_sdk::internal::flashlight_decoder {

class WordDictionary {
 public:
  WordDictionary() = default;
  explicit WordDictionary(const std::filesystem::path& path);

  int Size() const { return static_cast<int>(id_to_word_.size()); }
  bool Contains(const std::string& word) const;
  bool ContainsId(int id) const;
  int Id(const std::string& word) const;
  const std::string& Word(int id) const;
  const std::vector<std::string>& Entries() const { return id_to_word_; }

 private:
  std::vector<std::string> id_to_word_;
  std::unordered_map<std::string, int> word_to_id_;
};

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_WORD_DICTIONARY_H_
