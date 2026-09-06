#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_COMPACT_LEXICON_FORMAT_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_COMPACT_LEXICON_FORMAT_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace asr_sdk::internal::flashlight_decoder::compact_format {

constexpr uint8_t kMagic[8] = {'W', 'C', 'L', 'X', 'B', 'I', 'N', 1};
constexpr uint32_t kVersion = 1;
constexpr uint32_t kEndianMarker = 0x01020304U;
constexpr size_t kHeaderSize = 256;
constexpr size_t kSectionAlignment = 64;

enum Section : size_t {
  kEdgeOffsets = 0,
  kEdgeTokens = 1,
  kLabelOffsets = 2,
  kLabels = 3,
  kBaseLookahead = 4,
  kContextLookahead = 5,
  kWordPronunciationOffsets = 6,
  kPronunciationTokenOffsets = 7,
  kPronunciationTokens = 8,
  kSectionCount = 9,
};

constexpr size_t kVersionOffset = 8;
constexpr size_t kHeaderSizeOffset = 12;
constexpr size_t kEndianOffset = 16;
constexpr size_t kFlagsOffset = 20;
constexpr size_t kTokenCountOffset = 24;
constexpr size_t kWordCountOffset = 28;
constexpr size_t kNodeCountOffset = 32;
constexpr size_t kEdgeCountOffset = 36;
constexpr size_t kLabelCountOffset = 40;
constexpr size_t kPronunciationCountOffset = 44;
constexpr size_t kPronunciationTokenCountOffset = 48;
constexpr size_t kLexiconEntryCountOffset = 52;
constexpr size_t kSilenceIdOffset = 56;
constexpr size_t kBlankIdOffset = 60;
constexpr size_t kUnknownWordIdOffset = 64;
constexpr size_t kDependencyHashOffset = 72;
constexpr size_t kPayloadHashOffset = 80;
constexpr size_t kFileSizeOffset = 88;
constexpr size_t kSectionTableOffset = 96;

inline size_t Align(size_t value) {
  return (value + kSectionAlignment - 1) & ~(kSectionAlignment - 1);
}

inline void PutU32(std::vector<uint8_t>* bytes, size_t offset,
                   uint32_t value) {
  if (offset + 4 > bytes->size()) throw std::out_of_range("PutU32");
  for (int i = 0; i < 4; ++i) {
    (*bytes)[offset + i] = static_cast<uint8_t>(value >> (8 * i));
  }
}

inline void PutU64(std::vector<uint8_t>* bytes, size_t offset,
                   uint64_t value) {
  if (offset + 8 > bytes->size()) throw std::out_of_range("PutU64");
  for (int i = 0; i < 8; ++i) {
    (*bytes)[offset + i] = static_cast<uint8_t>(value >> (8 * i));
  }
}

inline uint32_t GetU32(const uint8_t* bytes, size_t size, size_t offset) {
  if (offset + 4 > size) throw std::runtime_error("truncated compact lexicon header");
  uint32_t value = 0;
  for (int i = 0; i < 4; ++i) {
    value |= static_cast<uint32_t>(bytes[offset + i]) << (8 * i);
  }
  return value;
}

inline uint64_t GetU64(const uint8_t* bytes, size_t size, size_t offset) {
  if (offset + 8 > size) throw std::runtime_error("truncated compact lexicon header");
  uint64_t value = 0;
  for (int i = 0; i < 8; ++i) {
    value |= static_cast<uint64_t>(bytes[offset + i]) << (8 * i);
  }
  return value;
}

inline uint64_t Fnv1a(const uint8_t* data, size_t size,
                      uint64_t hash = 14695981039346656037ULL) {
  for (size_t i = 0; i < size; ++i) {
    hash ^= data[i];
    hash *= 1099511628211ULL;
  }
  return hash;
}

inline uint64_t HashString(const std::string& value, uint64_t hash) {
  hash = Fnv1a(reinterpret_cast<const uint8_t*>(value.data()), value.size(), hash);
  const uint8_t separator = 0xff;
  return Fnv1a(&separator, 1, hash);
}

inline uint32_t FloatBits(float value) {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

}  // namespace asr_sdk::internal::flashlight_decoder::compact_format

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_COMPACT_LEXICON_FORMAT_H_
