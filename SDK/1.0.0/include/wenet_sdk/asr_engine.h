#ifndef WENET_SDK_ASR_ENGINE_H_
#define WENET_SDK_ASR_ENGINE_H_

#include <memory>

#include "wenet_sdk/config.h"
#include "wenet_sdk/stream.h"

namespace wenet_sdk {

class AsrEngine {
 public:
  static std::unique_ptr<AsrEngine> Create(const EngineConfig& config);

  virtual ~AsrEngine() = default;
  virtual std::unique_ptr<Stream> CreateStream() = 0;
};

}  // namespace wenet_sdk

#endif  // WENET_SDK_ASR_ENGINE_H_
