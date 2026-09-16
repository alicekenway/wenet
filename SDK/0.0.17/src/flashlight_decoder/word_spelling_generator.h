#ifndef ASR_SDK_WORD_SPELLING_GENERATOR_H_
#define ASR_SDK_WORD_SPELLING_GENERATOR_H_

#include <filesystem>
#include <string>
#include <unordered_map>
#include <vector>

#include "asr_sdk/status.h"
#include "sentencepiece_processor.h"
#include "sherpa_onnx_wenet/token_table.h"

namespace asr_sdk::internal::flashlight_decoder {

using TokenPaths = std::vector<std::vector<int>>;
struct GeneratedWordSpellings {
  TokenPaths paths;
  std::vector<std::string> diagnostics;
};

// Immutable after construction; SentencePiece's const Encode is thread-safe.
// Policy matches the static BPE builder: canonical, shortest, characters;
// ignore-case fallback, no standalone word boundary, at most three paths.
class WordSpellingGenerator {
 public:
  WordSpellingGenerator(const sherpa_onnx_wenet::TokenTable& tokens,
                        int blank_id, std::string boundary,
                        const std::filesystem::path& model);
  GeneratedWordSpellings Generate(const std::string& word) const;
  const std::vector<std::string>& Diagnostics() const { return diagnostics_; }

 private:
  int Resolve(const std::string& surface) const;
  sentencepiece::SentencePieceProcessor processor_;
  std::string boundary_;
  std::unordered_map<std::string, int> exact_;
  std::unordered_map<std::string, int> folded_;
  size_t max_piece_characters_ = 0;
  std::vector<std::string> diagnostics_;
};

// Input word paths are already deduplicated. Resource errors are preconditions,
// not unencodable forms: callers must never skip/truncate them.
StatusOr<TokenPaths> CombineWordSpellings(
    const std::vector<const GeneratedWordSpellings*>& words,
    size_t max_paths, size_t max_tokens);

}  // namespace asr_sdk::internal::flashlight_decoder
#endif
