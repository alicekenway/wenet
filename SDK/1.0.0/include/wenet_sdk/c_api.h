#ifndef WENET_SDK_C_API_H_
#define WENET_SDK_C_API_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct WenetSdkEngine WenetSdkEngine;
typedef struct WenetSdkStream WenetSdkStream;

typedef enum WenetSdkStatus {
  WENET_SDK_STATUS_OK = 0,
  WENET_SDK_STATUS_INVALID_ARGUMENT = 1,
  WENET_SDK_STATUS_NOT_FOUND = 2,
  WENET_SDK_STATUS_INTERNAL = 3,
  WENET_SDK_STATUS_UNAVAILABLE = 4
} WenetSdkStatus;

WenetSdkEngine* wenet_sdk_create_engine(const char* model_dir);
void wenet_sdk_destroy_engine(WenetSdkEngine* engine);

WenetSdkStream* wenet_sdk_create_stream(WenetSdkEngine* engine);
void wenet_sdk_destroy_stream(WenetSdkStream* stream);

int wenet_sdk_accept_pcm16(WenetSdkStream* stream, int sample_rate,
                           const int16_t* samples, int num_samples);
int wenet_sdk_accept_float32(WenetSdkStream* stream, int sample_rate,
                             const float* samples, int num_samples);

int wenet_sdk_decode(WenetSdkStream* stream);
int wenet_sdk_decode_ready(WenetSdkStream* stream);

const char* wenet_sdk_get_result_json(WenetSdkStream* stream);
const char* wenet_sdk_get_final_result_json(WenetSdkStream* stream);

void wenet_sdk_set_input_finished(WenetSdkStream* stream);
void wenet_sdk_reset_stream(WenetSdkStream* stream);

int wenet_sdk_last_error_code(void* handle);
const char* wenet_sdk_last_error_message(void* handle);

const char* wenet_sdk_version_string(void);
int wenet_sdk_abi_version(void);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // WENET_SDK_C_API_H_
