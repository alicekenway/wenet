#include "flashlight_decoder/debug_trace.h"

#include <iomanip>
#include <sstream>

namespace asr_sdk::internal::flashlight_decoder {
namespace {

std::string JsonEscape(const std::string& input) {
  std::string out;
  out.reserve(input.size());
  for (unsigned char c : input) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (c < 0x20) {
          const char* hex = "0123456789abcdef";
          out += "\\u00";
          out.push_back(hex[(c >> 4) & 0x0f]);
          out.push_back(hex[c & 0x0f]);
        } else {
          out.push_back(static_cast<char>(c));
        }
        break;
    }
  }
  return out;
}

std::string Quote(const std::string& value) {
  return "\"" + JsonEscape(value) + "\"";
}

void AppendHypothesisJson(const DecodedHypothesis& hyp, int rank,
                          std::ostringstream* out) {
  *out << "{\"rank\":" << rank;
  *out << ",\"text\":" << Quote(JoinWords(hyp.mapped_words, " "));
  *out << ",\"raw_text\":" << Quote(JoinWords(hyp.raw_words, " "));
  *out << ",\"am_mapped_text\":"
       << Quote(JoinWords(hyp.am_mapped_words, " "));
  *out << ",\"first_pass_score\":" << hyp.first_pass_score;
  *out << ",\"am_score\":" << hyp.am_score;
  *out << ",\"lm_score\":" << hyp.lm_score;
  *out << ",\"total_score\":" << hyp.total_score;
  *out << "}";
}

}  // namespace

std::string BuildDebugJson(const std::vector<std::string>& logs,
                           const std::string& error,
                           const std::vector<DecodedHypothesis>& hyps,
                           bool is_final) {
  std::ostringstream out;
  out << std::setprecision(10);
  out << "{\"debug\":true,\"mode\":\"shallow_fusion\",\"is_final\":"
      << (is_final ? "true" : "false");
  out << ",\"error\":" << Quote(error);
  out << ",\"logs\":[";
  for (size_t i = 0; i < logs.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << Quote(logs[i]);
  }
  out << "]";

  if (is_final) {
    out << ",\"final_nbest\":[";
    for (size_t i = 0; i < hyps.size(); ++i) {
      if (i != 0) {
        out << ",";
      }
      AppendHypothesisJson(hyps[i], static_cast<int>(i + 1), &out);
    }
    out << "]";
  } else if (!hyps.empty()) {
    out << ",\"partial_best\":";
    AppendHypothesisJson(hyps.front(), 1, &out);
  }

  out << "}";
  return out.str();
}

}  // namespace asr_sdk::internal::flashlight_decoder
