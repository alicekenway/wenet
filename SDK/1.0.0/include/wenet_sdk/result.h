#ifndef WENET_SDK_RESULT_H_
#define WENET_SDK_RESULT_H_

#include <string>
#include <vector>

namespace wenet_sdk {

struct TokenResult {
  std::string token;
  int token_id = -1;
  float start_ms = -1.0f;
  float end_ms = -1.0f;
  float confidence = 0.0f;
};

struct AsrResult {
  std::string text;
  bool is_final = false;
  float confidence = 0.0f;
  std::vector<TokenResult> tokens;
};

}  // namespace wenet_sdk

#endif  // WENET_SDK_RESULT_H_
