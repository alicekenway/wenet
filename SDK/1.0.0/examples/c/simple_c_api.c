#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "wenet_sdk/c_api.h"

int main(int argc, char** argv) {
  const char* model_dir = argc > 1 ? argv[1] : "model_example";
  WenetSdkEngine* engine = wenet_sdk_create_engine(model_dir);
  if (engine == NULL) {
    fprintf(stderr, "engine error: %s\n",
            wenet_sdk_last_error_message(NULL));
    return 1;
  }
  WenetSdkStream* stream = wenet_sdk_create_stream(engine);
  if (stream == NULL) {
    fprintf(stderr, "stream error: %s\n",
            wenet_sdk_last_error_message(engine));
    wenet_sdk_destroy_engine(engine);
    return 1;
  }

  const int sample_rate = 16000;
  const int num_samples = sample_rate / 2;
  float* pcm = (float*)calloc((size_t)num_samples, sizeof(float));
  if (pcm == NULL) {
    return 1;
  }
  for (int i = 0; i < num_samples; ++i) {
    pcm[i] = 0.2f * (float)sin(2.0 * 3.14159265358979323846 * 440.0 *
                               (double)i / sample_rate);
  }
  wenet_sdk_accept_float32(stream, sample_rate, pcm, num_samples);
  wenet_sdk_set_input_finished(stream);
  while (wenet_sdk_decode_ready(stream)) {
    wenet_sdk_decode(stream);
  }
  printf("%s\n", wenet_sdk_get_final_result_json(stream));

  free(pcm);
  wenet_sdk_destroy_stream(stream);
  wenet_sdk_destroy_engine(engine);
  return 0;
}
