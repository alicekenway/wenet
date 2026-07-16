#include "wenet_bridge/wenet_result_mapper.h"

#include "utils/json.h"

namespace asr_sdk::internal {

AsrResult MapWenetJsonResult(const std::string& json) {
  AsrResult result;
  result.raw_backend_json = json;
  result.is_final = json.find("final_result") != std::string::npos;

  const std::vector<std::string> sentences =
      ExtractJsonStringValues(json, "sentence");
  for (const auto& sentence : sentences) {
    NBestResult nbest;
    nbest.text = sentence;
    result.nbest.push_back(std::move(nbest));
  }
  if (!result.nbest.empty()) {
    result.text = result.nbest.front().text;
  }

  const std::vector<std::string> words = ExtractJsonStringValues(json, "word");
  result.tokens.reserve(words.size());
  for (const auto& word : words) {
    TokenResult token;
    token.token = word;
    result.tokens.push_back(std::move(token));
  }
  return result;
}

}  // namespace asr_sdk::internal
