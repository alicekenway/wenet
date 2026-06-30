#ifndef ASR_SDK_SRC_WENET_BRIDGE_WENET_SHARED_H_
#define ASR_SDK_SRC_WENET_BRIDGE_WENET_SHARED_H_

#include <memory>

#include "decoder/asr_decoder.h"
#include "frontend/feature_pipeline.h"

namespace asr_sdk::internal::wenet_types {

struct Shared {
  std::shared_ptr<wenet::FeaturePipelineConfig> feature_config;
  std::shared_ptr<wenet::DecodeResource> resource;
  std::shared_ptr<wenet::DecodeOptions> decode_options;
};

}  // namespace asr_sdk::internal::wenet_types

#endif  // ASR_SDK_SRC_WENET_BRIDGE_WENET_SHARED_H_
