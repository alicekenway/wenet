#ifndef ASR_SDK_SRC_ITN_ITN_PROCESSOR_H_
#define ASR_SDK_SRC_ITN_ITN_PROCESSOR_H_

#include <memory>
#include <string>

#include "asr_sdk/result.h"
#include "asr_sdk/status.h"
#include "asr_sdk/stream.h"

namespace wetext { class Processor; }

namespace asr_sdk::internal {

class ItnProcessor {
 public:
  ItnProcessor(const std::string& tagger, const std::string& verbalizer);
  ~ItnProcessor();
  std::string Normalize(const std::string& text) const;

 private:
  std::unique_ptr<wetext::Processor> processor_;
};

StatusOr<std::shared_ptr<const ItnProcessor>> CreateItnProcessor(
    const std::string& tagger, const std::string& verbalizer);
std::unique_ptr<AsrStream> WrapWithItn(
    std::unique_ptr<AsrStream> stream,
    std::shared_ptr<const ItnProcessor> processor);

}  // namespace asr_sdk::internal
#endif
