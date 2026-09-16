#include "flashlight_decoder/compact_lexicon.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include "flashlight_decoder/compact_lexicon_format.h"
#include "package/model_package.h"

namespace asr_sdk::internal::flashlight_decoder {
namespace {

using namespace compact_format;

struct SectionView {
  uint64_t offset = 0;
  uint64_t size = 0;
};

uint64_t CheckedBytes(uint64_t count, uint64_t width,
                      const char* section) {
  if (count > std::numeric_limits<uint64_t>::max() / width) {
    throw std::runtime_error(std::string("compact lexicon ") + section +
                             " size overflows");
  }
  return count * width;
}

void ValidateOffsetArray(const uint32_t* offsets, uint32_t count,
                         uint32_t expected_last, const char* label) {
  if (offsets[0] != 0) {
    throw std::runtime_error(std::string("compact lexicon ") + label +
                             " must start at zero");
  }
  for (uint32_t i = 0; i < count; ++i) {
    if (offsets[i] > offsets[i + 1] || offsets[i + 1] > expected_last) {
      throw std::runtime_error(std::string("compact lexicon invalid ") +
                               label);
    }
  }
  if (offsets[count] != expected_last) {
    throw std::runtime_error(std::string("compact lexicon ") + label +
                             " has the wrong final offset");
  }
}

uint64_t HashFile(const std::filesystem::path& path, const std::string& role,
                  uint64_t hash) {
  hash = HashString(role, hash);
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to hash compact lexicon dependency: " +
                             path.string());
  }
  std::array<char, 65536> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize count = input.gcount();
    if (count > 0) {
      hash = Fnv1a(reinterpret_cast<const uint8_t*>(buffer.data()),
                   static_cast<size_t>(count), hash);
    }
  }
  if (!input.eof()) {
    throw std::runtime_error("failed while hashing compact lexicon dependency: " +
                             path.string());
  }
  const uint8_t separator = 0xfe;
  return Fnv1a(&separator, 1, hash);
}

}  // namespace

struct CompactLexicon::Storage {
  ~Storage() {
    if (mapping != MAP_FAILED && mapping != nullptr) {
      munmap(mapping, size);
    }
  }

  const uint8_t* Data() const {
    return mapping != MAP_FAILED && mapping != nullptr
               ? static_cast<const uint8_t*>(mapping)
               : owned.data();
  }

  void* mapping = MAP_FAILED;
  size_t size = 0;
  std::vector<uint8_t> owned;
};

CompactLexicon::CompactLexicon(std::shared_ptr<Storage> storage)
    : storage_(std::move(storage)),
      file_data_(storage_->Data()),
      file_size_(storage_->size) {}

std::shared_ptr<const CompactLexicon> CompactLexicon::Load(
    const std::filesystem::path& path, uint64_t expected_dependency_hash) {
  const int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd < 0) {
    throw std::runtime_error("failed to open compact lexicon: " +
                             path.string() + ": " + std::strerror(errno));
  }
  struct stat stat_buffer {};
  if (fstat(fd, &stat_buffer) != 0 || stat_buffer.st_size < 0) {
    const int error = errno;
    close(fd);
    throw std::runtime_error("failed to stat compact lexicon: " +
                             path.string() + ": " + std::strerror(error));
  }
  auto storage = std::make_shared<Storage>();
  storage->size = static_cast<size_t>(stat_buffer.st_size);
  if (storage->size > 0) {
    storage->mapping = mmap(nullptr, storage->size, PROT_READ, MAP_PRIVATE, fd, 0);
  }
  close(fd);

  if (storage->mapping == MAP_FAILED) {
    storage->owned.resize(storage->size);
    std::ifstream input(path, std::ios::binary);
    if (!input ||
        (storage->size > 0 &&
         !input.read(reinterpret_cast<char*>(storage->owned.data()),
                     static_cast<std::streamsize>(storage->size)))) {
      throw std::runtime_error("failed to read compact lexicon: " +
                               path.string());
    }
  }
  auto lexicon = std::shared_ptr<CompactLexicon>(
      new CompactLexicon(std::move(storage)));
  lexicon->ParseAndValidate(expected_dependency_hash);
  return std::shared_ptr<const CompactLexicon>(std::move(lexicon));
}

void CompactLexicon::ParseAndValidate(uint64_t expected_dependency_hash) {
  if (file_size_ < kHeaderSize) {
    throw std::runtime_error("compact lexicon is smaller than its header");
  }
  if (std::memcmp(file_data_, kMagic, sizeof(kMagic)) != 0) {
    throw std::runtime_error("compact lexicon has invalid magic");
  }
  if (GetU32(file_data_, file_size_, kVersionOffset) != kVersion) {
    throw std::runtime_error("unsupported compact lexicon version");
  }
  if (GetU32(file_data_, file_size_, kHeaderSizeOffset) != kHeaderSize) {
    throw std::runtime_error("compact lexicon has invalid header size");
  }
  if (GetU32(file_data_, file_size_, kEndianOffset) != kEndianMarker) {
    throw std::runtime_error("compact lexicon has invalid endian marker");
  }
  const uint16_t host_endian = 1;
  if (*reinterpret_cast<const uint8_t*>(&host_endian) != 1) {
    throw std::runtime_error("compact lexicon reader requires little endian");
  }
  if (GetU32(file_data_, file_size_, kFlagsOffset) != 0) {
    throw std::runtime_error("compact lexicon uses unsupported flags");
  }

  token_count_ = GetU32(file_data_, file_size_, kTokenCountOffset);
  word_count_ = GetU32(file_data_, file_size_, kWordCountOffset);
  node_count_ = GetU32(file_data_, file_size_, kNodeCountOffset);
  edge_count_ = GetU32(file_data_, file_size_, kEdgeCountOffset);
  label_count_ = GetU32(file_data_, file_size_, kLabelCountOffset);
  pronunciation_count_ =
      GetU32(file_data_, file_size_, kPronunciationCountOffset);
  pronunciation_token_count_ =
      GetU32(file_data_, file_size_, kPronunciationTokenCountOffset);
  lexicon_entry_count_ =
      GetU32(file_data_, file_size_, kLexiconEntryCountOffset);
  silence_id_ = static_cast<int>(
      GetU32(file_data_, file_size_, kSilenceIdOffset));
  blank_id_ =
      static_cast<int>(GetU32(file_data_, file_size_, kBlankIdOffset));
  unknown_word_id_ = static_cast<int>(
      GetU32(file_data_, file_size_, kUnknownWordIdOffset));
  dependency_hash_ = GetU64(file_data_, file_size_, kDependencyHashOffset);
  payload_hash_ = GetU64(file_data_, file_size_, kPayloadHashOffset);
  const uint64_t declared_file_size =
      GetU64(file_data_, file_size_, kFileSizeOffset);

  if (declared_file_size != file_size_) {
    throw std::runtime_error("compact lexicon file-size field does not match");
  }
  if (expected_dependency_hash != 0 &&
      expected_dependency_hash != dependency_hash_) {
    throw std::runtime_error(
        "compact lexicon dependencies do not match the package (expected " +
        CompactLexiconHashHex(expected_dependency_hash) + ", found " +
        CompactLexiconHashHex(dependency_hash_) + ")");
  }
  if (Fnv1a(file_data_ + kHeaderSize, file_size_ - kHeaderSize) !=
      payload_hash_) {
    throw std::runtime_error("compact lexicon payload checksum mismatch");
  }
  if (token_count_ == 0 || token_count_ > UINT16_MAX || word_count_ == 0 ||
      node_count_ == 0 || edge_count_ != node_count_ - 1 ||
      pronunciation_count_ != lexicon_entry_count_) {
    throw std::runtime_error("compact lexicon has inconsistent counts");
  }
  if (silence_id_ < 0 || static_cast<uint32_t>(silence_id_) >= token_count_ ||
      blank_id_ < 0 || static_cast<uint32_t>(blank_id_) >= token_count_ ||
      unknown_word_id_ < 0 ||
      static_cast<uint32_t>(unknown_word_id_) >= word_count_) {
    throw std::runtime_error("compact lexicon has invalid special IDs");
  }

  std::array<SectionView, kSectionCount> sections{};
  std::array<uint64_t, kSectionCount> expected_sizes{
      CheckedBytes(static_cast<uint64_t>(node_count_) + 1, 4, "edge offsets"),
      CheckedBytes(edge_count_, 2, "edge tokens"),
      CheckedBytes(static_cast<uint64_t>(node_count_) + 1, 4, "label offsets"),
      CheckedBytes(label_count_, 4, "labels"),
      CheckedBytes(node_count_, 4, "base lookahead"),
      CheckedBytes(node_count_, 4, "context lookahead"),
      CheckedBytes(static_cast<uint64_t>(word_count_) + 1, 4,
                   "word pronunciation offsets"),
      CheckedBytes(static_cast<uint64_t>(pronunciation_count_) + 1, 4,
                   "pronunciation token offsets"),
      CheckedBytes(pronunciation_token_count_, 2, "pronunciation tokens")};
  uint64_t previous_end = kHeaderSize;
  for (size_t index = 0; index < kSectionCount; ++index) {
    const size_t descriptor = kSectionTableOffset + index * 16;
    sections[index].offset = GetU64(file_data_, file_size_, descriptor);
    sections[index].size = GetU64(file_data_, file_size_, descriptor + 8);
    if (sections[index].offset % kSectionAlignment != 0 ||
        sections[index].size != expected_sizes[index] ||
        sections[index].offset < previous_end ||
        sections[index].offset > file_size_ ||
        sections[index].size > file_size_ - sections[index].offset) {
      throw std::runtime_error("compact lexicon has an invalid section table");
    }
    previous_end = sections[index].offset + sections[index].size;
  }
  if (previous_end != file_size_) {
    throw std::runtime_error("compact lexicon has trailing data");
  }

  edge_offsets_ = reinterpret_cast<const uint32_t*>(
      file_data_ + sections[kEdgeOffsets].offset);
  edge_token_ids_ = reinterpret_cast<const uint16_t*>(
      file_data_ + sections[kEdgeTokens].offset);
  label_offsets_ = reinterpret_cast<const uint32_t*>(
      file_data_ + sections[kLabelOffsets].offset);
  label_word_ids_ = reinterpret_cast<const uint32_t*>(
      file_data_ + sections[kLabels].offset);
  base_lookahead_ = reinterpret_cast<const float*>(
      file_data_ + sections[kBaseLookahead].offset);
  context_lookahead_ = reinterpret_cast<const float*>(
      file_data_ + sections[kContextLookahead].offset);
  word_pronunciation_offsets_ = reinterpret_cast<const uint32_t*>(
      file_data_ + sections[kWordPronunciationOffsets].offset);
  pronunciation_token_offsets_ = reinterpret_cast<const uint32_t*>(
      file_data_ + sections[kPronunciationTokenOffsets].offset);
  pronunciation_token_ids_ = reinterpret_cast<const uint16_t*>(
      file_data_ + sections[kPronunciationTokens].offset);

  ValidateOffsetArray(edge_offsets_, node_count_, edge_count_, "edge offsets");
  ValidateOffsetArray(label_offsets_, node_count_, label_count_, "label offsets");
  ValidateOffsetArray(word_pronunciation_offsets_, word_count_,
                      pronunciation_count_, "word pronunciation offsets");
  ValidateOffsetArray(pronunciation_token_offsets_, pronunciation_count_,
                      pronunciation_token_count_,
                      "pronunciation token offsets");

  for (uint32_t state = 0; state < node_count_; ++state) {
    uint16_t previous_token = 0;
    bool first = true;
    for (uint32_t edge = edge_offsets_[state];
         edge < edge_offsets_[state + 1]; ++edge) {
      const uint16_t token = edge_token_ids_[edge];
      if (token >= token_count_ || (!first && token <= previous_token) ||
          edge + 1 >= node_count_) {
        throw std::runtime_error(
            "compact lexicon edges are invalid or not strictly sorted");
      }
      first = false;
      previous_token = token;
    }
    for (uint32_t label = label_offsets_[state];
         label < label_offsets_[state + 1]; ++label) {
      if (label_word_ids_[label] >= word_count_) {
        throw std::runtime_error("compact lexicon label word ID is invalid");
      }
    }
    if (std::isnan(base_lookahead_[state]) ||
        std::isnan(context_lookahead_[state]) ||
        base_lookahead_[state] == std::numeric_limits<float>::infinity() ||
        context_lookahead_[state] == std::numeric_limits<float>::infinity()) {
      throw std::runtime_error("compact lexicon contains an invalid score");
    }
  }
  for (uint32_t index = 0; index < pronunciation_token_count_; ++index) {
    if (pronunciation_token_ids_[index] >= token_count_) {
      throw std::runtime_error(
          "compact lexicon pronunciation token ID is invalid");
    }
  }
}

void CompactLexicon::RequireState(StateId state) const {
  if (state >= node_count_) {
    throw std::out_of_range("compact lexicon state is invalid");
  }
}

CompactLexicon::StateId CompactLexicon::Child(StateId state, int token) const {
  if (state == kInvalidState || token < 0 ||
      static_cast<uint32_t>(token) >= token_count_) {
    return kInvalidState;
  }
  RequireState(state);
  const uint32_t begin = edge_offsets_[state];
  const uint32_t end = edge_offsets_[state + 1];
  if (end - begin <= 8) {
    for (uint32_t edge = begin; edge < end; ++edge) {
      if (edge_token_ids_[edge] == token) return edge + 1;
      if (edge_token_ids_[edge] > token) break;
    }
    return kInvalidState;
  }
  const auto* found = std::lower_bound(edge_token_ids_ + begin,
                                       edge_token_ids_ + end,
                                       static_cast<uint16_t>(token));
  if (found == edge_token_ids_ + end || *found != token) {
    return kInvalidState;
  }
  return static_cast<StateId>((found - edge_token_ids_) + 1);
}

bool CompactLexicon::HasChildren(StateId state) const {
  if (state == kInvalidState) return false;
  RequireState(state);
  return edge_offsets_[state] != edge_offsets_[state + 1];
}

CompactArrayView<uint32_t> CompactLexicon::Labels(StateId state) const {
  if (state == kInvalidState) return {};
  RequireState(state);
  const uint32_t begin = label_offsets_[state];
  return CompactArrayView<uint32_t>(label_word_ids_ + begin,
                                    label_offsets_[state + 1] - begin);
}

float CompactLexicon::Lookahead(StateId state, bool context_mode) const {
  RequireState(state);
  return context_mode ? context_lookahead_[state] : base_lookahead_[state];
}

std::vector<std::vector<int>> CompactLexicon::WordSpellings(int word_id) const {
  if (word_id < 0 || static_cast<uint32_t>(word_id) >= word_count_) {
    return {};
  }
  const uint32_t begin = word_pronunciation_offsets_[word_id];
  const uint32_t end = word_pronunciation_offsets_[word_id + 1];
  std::vector<std::vector<int>> result;
  result.reserve(end - begin);
  for (uint32_t pronunciation = begin; pronunciation < end; ++pronunciation) {
    const uint32_t token_begin = pronunciation_token_offsets_[pronunciation];
    const uint32_t token_end = pronunciation_token_offsets_[pronunciation + 1];
    std::vector<int> tokens;
    tokens.reserve(token_end - token_begin);
    for (uint32_t index = token_begin; index < token_end; ++index) {
      tokens.push_back(pronunciation_token_ids_[index]);
    }
    result.push_back(std::move(tokens));
  }
  return result;
}

bool CompactLexicon::IsMemoryMapped() const {
  return storage_->mapping != MAP_FAILED && storage_->mapping != nullptr;
}

uint64_t ComputeCompactLexiconDependencyHash(const ModelPackage& package) {
  uint64_t hash = 14695981039346656037ULL;
  hash = HashString("compact_trie_v1", hash);
  hash = HashString(package.blank_token, hash);
  hash = HashString(package.sil_token, hash);
  hash = HashString(package.unk_word, hash);
  std::ostringstream settings;
  settings.imbue(std::locale::classic());
  settings << std::setprecision(17) << package.length_penalty << '\n'
           << package.flashlight_options.smearing << '\n';
  hash = HashString(settings.str(), hash);
  hash = HashFile(package.tokens_txt, "tokens.txt", hash);
  hash = HashFile(package.words_txt, "words.txt", hash);
  hash = HashFile(package.lm_search_json, "lm_search.json", hash);
  for (const FixedLmConfig& lm : package.fixed_lms) {
    hash = HashFile(lm.path, "lm:" + lm.filename, hash);
  }
  if (!package.output_mapping_txt.empty()) {
    hash = HashFile(package.output_mapping_txt, "output_mapping.txt", hash);
  } else {
    hash = HashString("output_mapping.txt:identity", hash);
  }
  if (!package.final_output_mapping_txt.empty()) {
    hash = HashFile(package.final_output_mapping_txt,
                    "final_output_mapping.txt", hash);
  } else {
    hash = HashString("final_output_mapping.txt:identity", hash);
  }
  return hash;
}

std::string CompactLexiconHashHex(uint64_t value) {
  std::ostringstream output;
  output << std::hex << std::setfill('0') << std::setw(16) << value;
  return output.str();
}

}  // namespace asr_sdk::internal::flashlight_decoder
