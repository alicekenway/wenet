#ifndef ASR_SDK_VERSION_H_
#define ASR_SDK_VERSION_H_

#include "asr_sdk/export.h"

namespace asr_sdk {

ASR_SDK_API const char* VersionString();
ASR_SDK_API int AbiVersion();
ASR_SDK_API const char* BuildInfoJson();

}  // namespace asr_sdk

#endif  // ASR_SDK_VERSION_H_
