#ifndef ASR_SDK_RESULT_H_
#define ASR_SDK_RESULT_H_

#include <string>
#include <vector>

namespace asr_sdk {

struct TokenResult {
  std::string token;
  int token_id = -1;
  float start_ms = -1.0f;
  float end_ms = -1.0f;
  float confidence = 0.0f;
};

struct NBestResult {
  std::string text;
  float score = 0.0f;
  std::vector<TokenResult> tokens;
};

struct AsrResult {
  std::string text;
  bool is_final = false;
  float confidence = 0.0f;
  std::vector<TokenResult> tokens;
  std::vector<NBestResult> nbest;
  std::string raw_backend_json;
};

}  // namespace asr_sdk

#endif  // ASR_SDK_RESULT_H_
