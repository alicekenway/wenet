#ifndef ASR_SDK_SRC_ANDROID_REDUCED_FST_LOG_H_
#define ASR_SDK_SRC_ANDROID_REDUCED_FST_LOG_H_

#include <stdexcept>

#define ASR_SDK_CHECK_IMPL(condition)                                      \
  do {                                                                     \
    if (!(condition)) {                                                    \
      throw std::runtime_error("check failed: " #condition);              \
    }                                                                      \
  } while (0)

#define CHECK(condition) ASR_SDK_CHECK_IMPL(condition)
#define CHECK_GE(a, b) ASR_SDK_CHECK_IMPL((a) >= (b))

#endif  // ASR_SDK_SRC_ANDROID_REDUCED_FST_LOG_H_
