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

struct ContactCandidateResult {
  std::string contact_id;
  std::string display_name;
  std::string value_id;
};

using SlotCandidateResult = ContactCandidateResult;

struct EntityResult {
  std::string type;
  std::string slot_token;
  std::string text;
  float start_ms = -1.0f;
  float end_ms = -1.0f;
  float score = 0.0f;
  bool ambiguous = false;
  std::vector<ContactCandidateResult> candidates;
};

struct NBestResult {
  std::string text;
  float score = 0.0f;
  std::vector<TokenResult> tokens;
  std::vector<EntityResult> entities;
};

struct AsrResult {
  std::string text;
  bool is_final = false;
  float confidence = 0.0f;
  std::vector<TokenResult> tokens;
  std::vector<EntityResult> entities;
  std::vector<NBestResult> nbest;
  std::string raw_backend_json;
};

}  // namespace asr_sdk

#endif  // ASR_SDK_RESULT_H_
