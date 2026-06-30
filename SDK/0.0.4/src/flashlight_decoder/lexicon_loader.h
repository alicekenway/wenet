#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_LEXICON_LOADER_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_LEXICON_LOADER_H_

#include <filesystem>
#include <string>
#include <vector>

#include "flashlight_decoder/word_dictionary.h"
#include "sherpa_onnx_wenet/token_table.h"

namespace asr_sdk::internal::flashlight_decoder {

struct LexiconEntry {
  int word_id = -1;
  std::string word;
  std::vector<int> token_ids;
  std::vector<std::string> tokens;
};

std::vector<LexiconEntry> LoadLexicon(
    const std::filesystem::path& path, const WordDictionary& words,
    const sherpa_onnx_wenet::TokenTable& tokens);

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_LEXICON_LOADER_H_
