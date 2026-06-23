#ifndef WENET_SDK_VERSION_H_
#define WENET_SDK_VERSION_H_

#define WENET_SDK_ABI_VERSION 1

#ifndef WENETSDK_VERSION_MAJOR
#define WENETSDK_VERSION_MAJOR 0
#endif

#ifndef WENETSDK_VERSION_MINOR
#define WENETSDK_VERSION_MINOR 1
#endif

#ifndef WENETSDK_VERSION_PATCH
#define WENETSDK_VERSION_PATCH 0
#endif

namespace wenet_sdk {

const char* VersionString();
int AbiVersion();

}  // namespace wenet_sdk

#endif  // WENET_SDK_VERSION_H_
