#ifndef WENET_SDK_SRC_CORE_RESULT_CONVERSION_H_
#define WENET_SDK_SRC_CORE_RESULT_CONVERSION_H_

#include "decoder/decoder_interface.h"
#include "decoder/symbol_table.h"
#include "postprocess/text_normalizer.h"
#include "postprocess/timestamp_estimator.h"
#include "wenet_sdk/result.h"

namespace wenet_sdk::internal {

class ResultBuilder {
 public:
  ResultBuilder(const SymbolTable* tokens, const SymbolTable* words,
                TextNormalizer normalizer,
                TimestampEstimator timestamp_estimator,
                bool enable_timestamps);

  AsrResult Build(const DecodeResult& decode_result, bool is_final) const;

 private:
  const SymbolTable* tokens_ = nullptr;
  const SymbolTable* words_ = nullptr;
  TextNormalizer normalizer_;
  TimestampEstimator timestamp_estimator_;
  bool enable_timestamps_ = false;
};

std::string AsrResultToJson(const AsrResult& result);

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_CORE_RESULT_CONVERSION_H_
