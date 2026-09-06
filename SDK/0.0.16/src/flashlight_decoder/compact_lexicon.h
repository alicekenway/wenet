#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_COMPACT_LEXICON_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_COMPACT_LEXICON_H_

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace asr_sdk::internal {
struct ModelPackage;
}

namespace asr_sdk::internal::flashlight_decoder {

template <typename T>
class CompactArrayView {
 public:
  CompactArrayView() = default;
  CompactArrayView(const T* data, uint32_t size) : data_(data), size_(size) {}

  const T* begin() const { return data_; }
  const T* end() const { return size_ == 0 ? data_ : data_ + size_; }
  const T& operator[](uint32_t index) const { return data_[index]; }
  uint32_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

 private:
  const T* data_ = nullptr;
  uint32_t size_ = 0;
};

// Read-only, pointer-free representation of a static token-to-word trie.
// State zero is the root and UINT32_MAX is the absent-state sentinel.
class CompactLexicon {
 public:
  using StateId = uint32_t;
  static constexpr StateId kRoot = 0;
  static constexpr StateId kInvalidState = UINT32_MAX;

  static std::shared_ptr<const CompactLexicon> Load(
      const std::filesystem::path& path,
      uint64_t expected_dependency_hash = 0);

  StateId Child(StateId state, int token) const;
  bool HasChildren(StateId state) const;
  CompactArrayView<uint32_t> Labels(StateId state) const;
  float Lookahead(StateId state, bool context_mode) const;
  std::vector<std::vector<int>> WordSpellings(int word_id) const;

  uint32_t TokenCount() const { return token_count_; }
  uint32_t WordCount() const { return word_count_; }
  uint32_t NodeCount() const { return node_count_; }
  uint32_t EdgeCount() const { return edge_count_; }
  uint32_t LabelCount() const { return label_count_; }
  uint32_t PronunciationCount() const { return pronunciation_count_; }
  uint32_t PronunciationTokenCount() const {
    return pronunciation_token_count_;
  }
  uint32_t LexiconEntryCount() const { return lexicon_entry_count_; }
  int SilenceId() const { return silence_id_; }
  int BlankId() const { return blank_id_; }
  int UnknownWordId() const { return unknown_word_id_; }
  uint64_t DependencyHash() const { return dependency_hash_; }
  uint64_t PayloadHash() const { return payload_hash_; }
  size_t FileSize() const { return file_size_; }
  bool IsMemoryMapped() const;

 private:
  struct Storage;
  explicit CompactLexicon(std::shared_ptr<Storage> storage);
  void ParseAndValidate(uint64_t expected_dependency_hash);
  void RequireState(StateId state) const;

  std::shared_ptr<Storage> storage_;
  const uint8_t* file_data_ = nullptr;
  size_t file_size_ = 0;

  uint32_t token_count_ = 0;
  uint32_t word_count_ = 0;
  uint32_t node_count_ = 0;
  uint32_t edge_count_ = 0;
  uint32_t label_count_ = 0;
  uint32_t pronunciation_count_ = 0;
  uint32_t pronunciation_token_count_ = 0;
  uint32_t lexicon_entry_count_ = 0;
  int silence_id_ = -1;
  int blank_id_ = -1;
  int unknown_word_id_ = -1;
  uint64_t dependency_hash_ = 0;
  uint64_t payload_hash_ = 0;

  const uint32_t* edge_offsets_ = nullptr;
  const uint16_t* edge_token_ids_ = nullptr;
  const uint32_t* label_offsets_ = nullptr;
  const uint32_t* label_word_ids_ = nullptr;
  const float* base_lookahead_ = nullptr;
  const float* context_lookahead_ = nullptr;
  const uint32_t* word_pronunciation_offsets_ = nullptr;
  const uint32_t* pronunciation_token_offsets_ = nullptr;
  const uint16_t* pronunciation_token_ids_ = nullptr;
};

// Hashes all package files and settings that affect word IDs or precomputed
// lookahead. The source lexicon itself is protected by the binary payload hash.
uint64_t ComputeCompactLexiconDependencyHash(const ModelPackage& package);

std::string CompactLexiconHashHex(uint64_t value);

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_COMPACT_LEXICON_H_
