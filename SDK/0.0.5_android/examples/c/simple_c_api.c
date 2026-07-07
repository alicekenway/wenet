#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "asr_sdk/c_api.h"

static int read_all(const char* path, int16_t** out_data, int* out_samples) {
  FILE* f = fopen(path, "rb");
  long size = 0;
  if (!f) return 0;
  fseek(f, 0, SEEK_END);
  size = ftell(f);
  fseek(f, 44, SEEK_SET);
  if (size <= 44) {
    fclose(f);
    return 0;
  }
  *out_samples = (int)((size - 44) / 2);
  *out_data = (int16_t*)malloc((size_t)(*out_samples) * sizeof(int16_t));
  if (!*out_data) {
    fclose(f);
    return 0;
  }
  fread(*out_data, sizeof(int16_t), (size_t)*out_samples, f);
  fclose(f);
  return 1;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr, "usage: %s MODEL_DIR WAV\n", argv[0]);
    return 2;
  }
  AsrSdkEngine* engine = NULL;
  AsrSdkStream* stream = NULL;
  int rc = asr_sdk_create_engine(argv[1], &engine);
  if (rc != 0) {
    fprintf(stderr, "%s\n", asr_sdk_last_error_message(NULL));
    return 1;
  }
  rc = asr_sdk_create_stream(engine, &stream);
  if (rc != 0) {
    fprintf(stderr, "%s\n", asr_sdk_last_error_message(engine));
    asr_sdk_destroy_engine(engine);
    return 1;
  }
  int16_t* samples = NULL;
  int num_samples = 0;
  if (!read_all(argv[2], &samples, &num_samples)) {
    fprintf(stderr, "failed to read wav payload\n");
    asr_sdk_destroy_stream(stream);
    asr_sdk_destroy_engine(engine);
    return 1;
  }
  rc = asr_sdk_accept_pcm16(stream, samples, num_samples, 16000);
  free(samples);
  if (rc == 0) rc = asr_sdk_set_input_finished(stream);
  while (rc == 0 && asr_sdk_decode_ready(stream)) {
    rc = asr_sdk_decode(stream);
  }
  if (rc != 0) {
    fprintf(stderr, "%s\n", asr_sdk_last_error_message(stream));
  } else {
    printf("%s\n", asr_sdk_get_final_result_json(stream));
  }
  asr_sdk_destroy_stream(stream);
  asr_sdk_destroy_engine(engine);
  return rc == 0 ? 0 : 1;
}
