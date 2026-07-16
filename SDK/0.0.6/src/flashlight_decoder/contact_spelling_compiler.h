#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_CONTACT_SPELLING_COMPILER_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_CONTACT_SPELLING_COMPILER_H_

#include <string>
#include <vector>

#include "asr_sdk/decode_context.h"
#include "asr_sdk/status.h"

namespace asr_sdk::internal::flashlight_decoder {

class FlashlightDecoderResource;

struct CompiledContactSpelling {
  std::string contact_id;
  std::string display_name;
  std::string spoken_form;
  std::vector<int> token_ids;
  int logical_word_count = 1;
};

// Converts runner-provided spellings to sequences from the existing AM token
// table.  It never adds or modifies acoustic-model tokens.
class ContactSpellingCompiler {
 public:
  static StatusOr<std::vector<CompiledContactSpelling>> Compile(
      const FlashlightDecoderResource& resource,
      const DecodeContextConfig& config, std::vector<std::string>* diagnostics);
};

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_CONTACT_SPELLING_COMPILER_H_
