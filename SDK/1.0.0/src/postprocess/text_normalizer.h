#ifndef WENET_SDK_SRC_POSTPROCESS_TEXT_NORMALIZER_H_
#define WENET_SDK_SRC_POSTPROCESS_TEXT_NORMALIZER_H_

#include <string>
#include <vector>

namespace wenet_sdk::internal {

struct TextNormalizerOptions {
  bool lowercase = true;
  bool remove_bpe_marker = true;
  std::string language_type = "indo_european";
};

class TextNormalizer {
 public:
  explicit TextNormalizer(TextNormalizerOptions options);

  std::string NormalizeTokens(const std::vector<std::string>& tokens) const;

 private:
  TextNormalizerOptions options_;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_POSTPROCESS_TEXT_NORMALIZER_H_
