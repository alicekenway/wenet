#include "asr_sdk/version.h"

#include <string>

namespace asr_sdk {
namespace {

#ifndef ASR_SDK_VERSION_STRING
#define ASR_SDK_VERSION_STRING "0.0.1"
#endif

#ifndef ASR_SDK_ABI_VERSION
#define ASR_SDK_ABI_VERSION 1
#endif

#ifndef ASR_SDK_WENET_COMMIT
#define ASR_SDK_WENET_COMMIT "unknown"
#endif

#ifndef ASR_SDK_ONNXRUNTIME_VERSION
#define ASR_SDK_ONNXRUNTIME_VERSION "unknown"
#endif

std::string MakeBuildInfo() {
  return std::string("{\"sdk_version\":\"") + ASR_SDK_VERSION_STRING +
         "\",\"abi_version\":" + std::to_string(ASR_SDK_ABI_VERSION) +
         ",\"wenet\":{\"commit\":\"" + ASR_SDK_WENET_COMMIT +
         "\",\"linkage\":\"static\"},\"onnxruntime\":{\"version\":\"" +
         ASR_SDK_ONNXRUNTIME_VERSION +
         "\",\"linkage\":\"dynamic\"},\"symbol_visibility\":\"hidden\"}";
}

}  // namespace

const char* VersionString() { return ASR_SDK_VERSION_STRING; }

int AbiVersion() { return ASR_SDK_ABI_VERSION; }

const char* BuildInfoJson() {
  static const std::string info = MakeBuildInfo();
  return info.c_str();
}

}  // namespace asr_sdk
