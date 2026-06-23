#include "postprocess/text_normalizer.h"

#include <algorithm>
#include <cctype>
#include <utility>

namespace wenet_sdk::internal {
namespace {

bool IsSpecialToken(const std::string& token) {
  return token.size() >= 2 && token.front() == '<' && token.back() == '>';
}

}  // namespace

TextNormalizer::TextNormalizer(TextNormalizerOptions options)
    : options_(std::move(options)) {}

std::string TextNormalizer::NormalizeTokens(
    const std::vector<std::string>& tokens) const {
  std::string text;
  for (std::string token : tokens) {
    if (token.empty() || IsSpecialToken(token)) {
      continue;
    }
    if (options_.remove_bpe_marker) {
      const std::string sentencepiece_marker = "\xe2\x96\x81";
      const auto sp = token.find(sentencepiece_marker);
      if (sp != std::string::npos) {
        token.replace(sp, sentencepiece_marker.size(), " ");
      }
      if (token.rfind("@@", token.size() >= 2 ? token.size() - 2 : 0) !=
          std::string::npos) {
        token.erase(token.size() - 2);
      }
    }
    if (!text.empty() && !token.empty() && token.front() != ' ' &&
        options_.language_type == "indo_european") {
      text.push_back(' ');
    }
    text += token;
  }
  if (options_.lowercase) {
    std::transform(text.begin(), text.end(), text.begin(),
                   [](unsigned char c) { return std::tolower(c); });
  }
  const auto begin = text.find_first_not_of(' ');
  if (begin == std::string::npos) {
    return "";
  }
  const auto end = text.find_last_not_of(' ');
  return text.substr(begin, end - begin + 1);
}

}  // namespace wenet_sdk::internal
