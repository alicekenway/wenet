#include "sdk/result_json.h"

#include <sstream>

#include "utils/json.h"

namespace asr_sdk::internal {

std::string AsrResultToJson(const AsrResult& result) {
  std::ostringstream out;
  out << "{\"text\":\"" << JsonEscape(result.text) << "\",";
  out << "\"is_final\":" << (result.is_final ? "true" : "false") << ",";
  out << "\"confidence\":" << result.confidence << ",";
  out << "\"nbest\":[";
  for (size_t i = 0; i < result.nbest.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << "{\"text\":\"" << JsonEscape(result.nbest[i].text)
        << "\",\"score\":" << result.nbest[i].score << "}";
  }
  out << "],\"tokens\":[";
  for (size_t i = 0; i < result.tokens.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << "{\"token\":\"" << JsonEscape(result.tokens[i].token)
        << "\",\"token_id\":" << result.tokens[i].token_id
        << ",\"start_ms\":" << result.tokens[i].start_ms
        << ",\"end_ms\":" << result.tokens[i].end_ms
        << ",\"confidence\":" << result.tokens[i].confidence << "}";
  }
  out << "]";
  if (!result.raw_backend_json.empty()) {
    out << ",\"raw_backend_json\":\""
        << JsonEscape(result.raw_backend_json) << "\"";
  }
  out << "}";
  return out.str();
}

}  // namespace asr_sdk::internal
