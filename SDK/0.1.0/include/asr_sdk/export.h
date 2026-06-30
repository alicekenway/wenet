#ifndef ASR_SDK_EXPORT_H_
#define ASR_SDK_EXPORT_H_

#if defined(_WIN32)
#if defined(ASR_SDK_BUILDING_LIBRARY)
#define ASR_SDK_API __declspec(dllexport)
#else
#define ASR_SDK_API __declspec(dllimport)
#endif
#else
#define ASR_SDK_API __attribute__((visibility("default")))
#endif

#endif  // ASR_SDK_EXPORT_H_
