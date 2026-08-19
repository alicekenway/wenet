#ifndef ASR_SDK_ASR_ENGINE_H_
#define ASR_SDK_ASR_ENGINE_H_

#include <memory>

#include "asr_sdk/config.h"
#include "asr_sdk/decode_context.h"
#include "asr_sdk/export.h"
#include "asr_sdk/status.h"
#include "asr_sdk/stream.h"

namespace asr_sdk {

class ASR_SDK_API AsrEngine {
 public:
  static StatusOr<std::unique_ptr<AsrEngine>> Create(
      const EngineConfig& config);

  virtual ~AsrEngine() = default;

  // Compiles application-owned runtime contacts into an immutable context.
  // The resulting context may only be used with this engine instance.
  virtual StatusOr<std::shared_ptr<const DecodeContext>> CompileDecodeContext(
      const DecodeContextConfig& config) = 0;

  // A null or empty context is equivalent to the compatibility overload.
  virtual StatusOr<std::unique_ptr<AsrStream>> CreateStream(
      std::shared_ptr<const DecodeContext> context) = 0;

  // Keep the base decoder path intact for callers that do not use contacts.
  virtual StatusOr<std::unique_ptr<AsrStream>> CreateStream() = 0;
  virtual const EngineConfig& config() const = 0;
};

}  // namespace asr_sdk

#endif  // ASR_SDK_ASR_ENGINE_H_
