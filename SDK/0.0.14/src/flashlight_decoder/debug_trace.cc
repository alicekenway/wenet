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

std::string DebugText(const DecodedHypothesis& hyp) {
  std::vector<std::string> output;
  for (size_t index = 0; index < hyp.mapped_words.size();) {
    const DecodedWord& word = hyp.mapped_words[index];
    if (!word.is_contact) {
      output.push_back(word.text);
      ++index;
      continue;
    }
    const int entity_index = word.contact_entity_index;
    std::string tagged;
    while (index < hyp.mapped_words.size() &&
           hyp.mapped_words[index].is_contact &&
           hyp.mapped_words[index].contact_entity_index == entity_index) {
      if (!tagged.empty()) {
        tagged.push_back('_');
      }
      tagged += hyp.mapped_words[index].text;
      ++index;
    }
    std::string slot_token = "<SLOT>";
    if (entity_index >= 0 &&
        entity_index < static_cast<int>(hyp.entities.size())) {
      slot_token = hyp.entities[static_cast<size_t>(entity_index)].slot_token;
    }
    output.push_back(tagged + slot_token);
  }
  std::string joined;
  for (const std::string& item : output) {
    if (!joined.empty()) {
      joined.push_back(' ');
    }
    joined += item;
  }
  return joined;
}

void AppendHypothesisJson(const DecodedHypothesis& hyp, int rank,
                          std::ostringstream* out) {
  *out << "{\"rank\":" << rank;
  *out << ",\"decoder_source\":"
       << Quote("max_fusion");
  *out << ",\"text\":" << Quote(JoinWords(hyp.mapped_words, " "));
  *out << ",\"debug_text\":" << Quote(DebugText(hyp));
  *out << ",\"raw_text\":" << Quote(JoinWords(hyp.raw_words, " "));
  *out << ",\"am_mapped_text\":"
       << Quote(JoinWords(hyp.am_mapped_words, " "));
  *out << ",\"first_pass_score\":" << hyp.first_pass_score;
  *out << ",\"am_score\":" << hyp.am_score;
  *out << ",\"lm_score\":" << hyp.lm_score;
  *out << ",\"weighted_lm_score\":" << hyp.weighted_lm_score;
  *out << ",\"total_score\":" << hyp.total_score;
  *out << ",\"entities\":[";
  for (size_t i = 0; i < hyp.entities.size(); ++i) {
    if (i != 0) {
      *out << ",";
    }
    const DecodedEntity& entity = hyp.entities[i];
    *out << "{\"type\":" << Quote(entity.type)
         << ",\"slot_token\":" << Quote(entity.slot_token)
         << ",\"text\":" << Quote(entity.text)
         << ",\"bias_score\":" << entity.score;
    if (entity.slot_token == "<CONTACT>") {
      *out << ",\"contact_lm_class_score\":" << entity.score;
    }
    *out
         << ",\"ambiguous\":" << (entity.ambiguous ? "true" : "false")
         << ",\"candidate_ids\":[";
    for (size_t j = 0; j < entity.candidates.size(); ++j) {
      if (j != 0) {
        *out << ",";
      }
      *out << Quote(entity.candidates[j].value_id);
    }
    *out << "]}";
  }
  *out << "]";
  *out << ",\"lm_events\":[";
  for (size_t i = 0; i < hyp.lm_events.size(); ++i) {
    if (i != 0) *out << ",";
    const LmEventTrace& event = hyp.lm_events[i];
    *out << "{\"event\":" << Quote(event.event)
         << ",\"filename\":" << Quote(event.filename)
         << ",\"type\":" << Quote(event.type)
         << ",\"raw_score\":" << event.raw_score
         << ",\"reference_score\":" << event.reference_score
         << ",\"adjusted_score\":" << event.adjusted_score
         << ",\"weighted_score\":" << event.weighted_score << "}";
  }
  *out << "]";
  *out << "}";
}

}  // namespace

std::string BuildDebugJson(const std::vector<std::string>& logs,
                           const std::string& error,
                           const std::vector<DecodedHypothesis>& hyps,
                           bool is_final) {
  std::ostringstream out;
  out << std::setprecision(10);
  out << "{\"debug\":true,\"mode\":\"max_fusion\",\"is_final\":"
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
