#include "core/result_conversion.h"

#include <sstream>
#include <utility>

#include "utils/json.h"

namespace wenet_sdk::internal {

ResultBuilder::ResultBuilder(const SymbolTable* tokens, const SymbolTable* words,
                             TextNormalizer normalizer,
                             TimestampEstimator timestamp_estimator,
                             bool enable_timestamps)
    : tokens_(tokens),
      words_(words),
      normalizer_(std::move(normalizer)),
      timestamp_estimator_(timestamp_estimator),
      enable_timestamps_(enable_timestamps) {}

AsrResult ResultBuilder::Build(const DecodeResult& decode_result,
                               bool is_final) const {
  AsrResult result;
  result.is_final = is_final;
  result.confidence = decode_result.confidence;

  std::vector<std::string> pieces;
  const std::vector<int>& ids =
      !decode_result.word_ids.empty() ? decode_result.word_ids
                                      : decode_result.token_ids;
  const SymbolTable* table =
      !decode_result.word_ids.empty() && words_ != nullptr ? words_ : tokens_;
  pieces.reserve(ids.size());

  for (size_t i = 0; i < ids.size(); ++i) {
    const int id = ids[i];
    const std::string symbol = table != nullptr ? table->Symbol(id)
                                                : std::to_string(id);
    pieces.push_back(symbol);

    TokenResult token;
    token.token = symbol;
    token.token_id = id;
    token.confidence = decode_result.confidence;
    if (enable_timestamps_ && i < decode_result.frame_indexes.size()) {
      token.start_ms =
          timestamp_estimator_.StartMs(decode_result.frame_indexes[i]);
      token.end_ms = timestamp_estimator_.EndMs(decode_result.frame_indexes[i]);
    }
    result.tokens.push_back(std::move(token));
  }

  result.text = normalizer_.NormalizeTokens(pieces);
  return result;
}

std::string AsrResultToJson(const AsrResult& result) {
  std::ostringstream os;
  os << "{\"text\":\"" << JsonEscape(result.text) << "\",";
  os << "\"is_final\":" << (result.is_final ? "true" : "false") << ",";
  os << "\"confidence\":" << result.confidence << ",";
  os << "\"tokens\":[";
  for (size_t i = 0; i < result.tokens.size(); ++i) {
    const auto& token = result.tokens[i];
    if (i > 0) {
      os << ",";
    }
    os << "{\"token\":\"" << JsonEscape(token.token) << "\",";
    os << "\"token_id\":" << token.token_id << ",";
    os << "\"start_ms\":" << token.start_ms << ",";
    os << "\"end_ms\":" << token.end_ms << ",";
    os << "\"confidence\":" << token.confidence << "}";
  }
  os << "]}";
  return os.str();
}

}  // namespace wenet_sdk::internal
