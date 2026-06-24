#ifndef ASR_SDK_C_API_H_
#define ASR_SDK_C_API_H_

#include <stdint.h>

#include "asr_sdk/export.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AsrSdkEngine AsrSdkEngine;
typedef struct AsrSdkStream AsrSdkStream;

typedef enum AsrSdkStatusCode {
  ASR_SDK_STATUS_OK = 0,
  ASR_SDK_STATUS_INVALID_ARGUMENT = 1,
  ASR_SDK_STATUS_NOT_FOUND = 2,
  ASR_SDK_STATUS_INTERNAL = 3,
  ASR_SDK_STATUS_UNAVAILABLE = 4,
  ASR_SDK_STATUS_FAILED_PRECONDITION = 5
} AsrSdkStatusCode;

ASR_SDK_API int asr_sdk_create_engine(const char* model_dir,
                                      AsrSdkEngine** out_engine);
ASR_SDK_API void asr_sdk_destroy_engine(AsrSdkEngine* engine);

ASR_SDK_API int asr_sdk_create_stream(AsrSdkEngine* engine,
                                      AsrSdkStream** out_stream);
ASR_SDK_API void asr_sdk_destroy_stream(AsrSdkStream* stream);

ASR_SDK_API int asr_sdk_accept_pcm16(AsrSdkStream* stream,
                                     const int16_t* samples, int num_samples,
                                     int sample_rate);
ASR_SDK_API int asr_sdk_decode(AsrSdkStream* stream);
ASR_SDK_API int asr_sdk_decode_ready(AsrSdkStream* stream);
ASR_SDK_API int asr_sdk_set_input_finished(AsrSdkStream* stream);
ASR_SDK_API int asr_sdk_reset_stream(AsrSdkStream* stream);

ASR_SDK_API const char* asr_sdk_get_result_json(AsrSdkStream* stream);
ASR_SDK_API const char* asr_sdk_get_final_result_json(AsrSdkStream* stream);

ASR_SDK_API int asr_sdk_last_error_code(void* handle);
ASR_SDK_API const char* asr_sdk_last_error_message(void* handle);

ASR_SDK_API const char* asr_sdk_version(void);
ASR_SDK_API int asr_sdk_abi_version(void);
ASR_SDK_API const char* asr_sdk_build_info_json(void);

#ifdef __cplusplus
}
#endif

#endif  // ASR_SDK_C_API_H_
