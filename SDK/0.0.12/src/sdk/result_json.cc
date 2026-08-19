#include "sdk/result_json.h"

#include <sstream>

#include "utils/json.h"

namespace asr_sdk::internal {
namespace {

void AppendEntities(std::ostringstream* out,
                    const std::vector<EntityResult>& entities) {
  *out << "[";
  for (size_t i = 0; i < entities.size(); ++i) {
    if (i > 0) {
      *out << ",";
    }
    const EntityResult& entity = entities[i];
    *out << "{\"type\":\"" << JsonEscape(entity.type)
         << "\",\"slot_token\":\"" << JsonEscape(entity.slot_token)
         << "\",\"text\":\"" << JsonEscape(entity.text)
         << "\",\"start_ms\":" << entity.start_ms
         << ",\"end_ms\":" << entity.end_ms
         << ",\"score\":" << entity.score
         << ",\"ambiguous\":" << (entity.ambiguous ? "true" : "false")
         << ",\"candidates\":[";
    for (size_t j = 0; j < entity.candidates.size(); ++j) {
      if (j > 0) {
        *out << ",";
      }
      const ContactCandidateResult& candidate = entity.candidates[j];
      *out << "{\"value_id\":\"" << JsonEscape(candidate.value_id)
           << "\",\"display_name\":\""
           << JsonEscape(candidate.display_name) << "\"";
      if (!candidate.contact_id.empty()) {
        *out << ",\"contact_id\":\"" << JsonEscape(candidate.contact_id)
             << "\"";
      }
      *out << "}";
    }
    *out << "]}";
  }
  *out << "]";
}

}  // namespace

std::string AsrResultToJson(const AsrResult& result) {
  std::ostringstream out;
  out << "{\"text\":\"" << JsonEscape(result.text) << "\",";
  if (!result.raw_text.empty()) {
    out << "\"raw_text\":\"" << JsonEscape(result.raw_text) << "\",";
  }
  out << "\"is_final\":" << (result.is_final ? "true" : "false") << ",";
  out << "\"confidence\":" << result.confidence << ",";
  out << "\"nbest\":[";
  for (size_t i = 0; i < result.nbest.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << "{\"text\":\"" << JsonEscape(result.nbest[i].text)
        << "\",\"score\":" << result.nbest[i].score
        << ",\"entities\":";
    AppendEntities(&out, result.nbest[i].entities);
    out << "}";
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
  out << "],\"entities\":";
  AppendEntities(&out, result.entities);
  if (!result.raw_backend_json.empty()) {
    out << ",\"raw_backend_json\":\""
        << JsonEscape(result.raw_backend_json) << "\"";
  }
  out << "}";
  return out.str();
}

}  // namespace asr_sdk::internal
