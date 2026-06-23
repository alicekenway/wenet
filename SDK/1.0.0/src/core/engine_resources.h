#ifndef WENET_SDK_SRC_CORE_ENGINE_RESOURCES_H_
#define WENET_SDK_SRC_CORE_ENGINE_RESOURCES_H_

#include "decoder/symbol_table.h"
#include "model/model_metadata.h"
#include "wenet_sdk/config.h"

namespace wenet_sdk::internal {

struct EngineResources {
  EngineConfig config;
  ModelMetadata metadata;
  SymbolTable tokens;
  SymbolTable words;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_CORE_ENGINE_RESOURCES_H_
