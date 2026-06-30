#ifndef ASR_SDK_ASR_ENGINE_H_
#define ASR_SDK_ASR_ENGINE_H_

#include <memory>

#include "asr_sdk/config.h"
#include "asr_sdk/export.h"
#include "asr_sdk/status.h"
#include "asr_sdk/stream.h"

namespace asr_sdk {

class ASR_SDK_API AsrEngine {
 public:
  static StatusOr<std::unique_ptr<AsrEngine>> Create(
      const EngineConfig& config);

  virtual ~AsrEngine() = default;
  virtual StatusOr<std::unique_ptr<AsrStream>> CreateStream() = 0;
  virtual const EngineConfig& config() const = 0;
};

}  // namespace asr_sdk

#endif  // ASR_SDK_ASR_ENGINE_H_
