#include <android/log.h>
#include <jni.h>

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "asr_sdk/c_api.h"

#define LOG_TAG "ASR_TEST"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

namespace {

struct WavData {
  int sample_rate = 0;
  std::vector<int16_t> samples;
};

uint32_t ReadU32(std::ifstream* in) {
  unsigned char b[4] = {0, 0, 0, 0};
  in->read(reinterpret_cast<char*>(b), 4);
  return static_cast<uint32_t>(b[0]) | (static_cast<uint32_t>(b[1]) << 8) |
         (static_cast<uint32_t>(b[2]) << 16) |
         (static_cast<uint32_t>(b[3]) << 24);
}

uint16_t ReadU16(std::ifstream* in) {
  unsigned char b[2] = {0, 0};
  in->read(reinterpret_cast<char*>(b), 2);
  return static_cast<uint16_t>(b[0]) | (static_cast<uint16_t>(b[1]) << 8);
}

std::string ReadTag(std::ifstream* in) {
  char tag[4] = {0, 0, 0, 0};
  in->read(tag, 4);
  return std::string(tag, 4);
}

bool ReadWavFile(const std::string& path, WavData* data, std::string* error) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    *error = "failed to open wav: " + path;
    return false;
  }
  if (ReadTag(&in) != "RIFF") {
    *error = "missing RIFF header";
    return false;
  }
  (void)ReadU32(&in);
  if (ReadTag(&in) != "WAVE") {
    *error = "missing WAVE header";
    return false;
  }

  uint16_t audio_format = 0;
  uint16_t channels = 0;
  uint32_t sample_rate = 0;
  uint16_t bits_per_sample = 0;
  std::vector<char> pcm_bytes;

  while (in && (!audio_format || pcm_bytes.empty())) {
    std::string tag = ReadTag(&in);
    if (!in) {
      break;
    }
    uint32_t size = ReadU32(&in);
    if (tag == "fmt ") {
      audio_format = ReadU16(&in);
      channels = ReadU16(&in);
      sample_rate = ReadU32(&in);
      (void)ReadU32(&in);
      (void)ReadU16(&in);
      bits_per_sample = ReadU16(&in);
      if (size > 16) {
        in.seekg(static_cast<std::streamoff>(size - 16), std::ios::cur);
      }
    } else if (tag == "data") {
      pcm_bytes.resize(size);
      in.read(pcm_bytes.data(), static_cast<std::streamsize>(size));
    } else {
      in.seekg(static_cast<std::streamoff>(size), std::ios::cur);
    }
    if (size % 2 == 1) {
      in.seekg(1, std::ios::cur);
    }
  }

  if (audio_format != 1 || bits_per_sample != 16 || channels == 0) {
    *error = "only 16-bit PCM WAV is supported";
    return false;
  }
  data->sample_rate = static_cast<int>(sample_rate);
  data->samples.clear();
  const size_t total_samples = pcm_bytes.size() / sizeof(int16_t);
  data->samples.reserve(total_samples / channels);
  for (size_t i = 0; i + channels <= total_samples; i += channels) {
    int32_t mix = 0;
    for (uint16_t c = 0; c < channels; ++c) {
      const size_t byte = (i + c) * sizeof(int16_t);
      const auto lo = static_cast<unsigned char>(pcm_bytes[byte]);
      const auto hi = static_cast<unsigned char>(pcm_bytes[byte + 1]);
      mix += static_cast<int16_t>(lo | static_cast<uint16_t>(hi << 8));
    }
    data->samples.push_back(static_cast<int16_t>(mix / channels));
  }
  return true;
}

std::string LastErrorJson(void* handle, const std::string& step) {
  const char* message = asr_sdk_last_error_message(handle);
  int code = asr_sdk_last_error_code(handle);
  return "{\"step\":\"" + step + "\",\"code\":" + std::to_string(code) +
         ",\"message\":\"" + (message ? message : "") + "\"}";
}

}  // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_asrdemo_AsrNative_runWavTest(JNIEnv* env, jclass,
                                              jstring modelDir,
                                              jstring wavPath) {
  const char* model_dir_chars = env->GetStringUTFChars(modelDir, nullptr);
  const char* wav_path_chars = env->GetStringUTFChars(wavPath, nullptr);
  std::string model_dir = model_dir_chars ? model_dir_chars : "";
  std::string wav_path = wav_path_chars ? wav_path_chars : "";
  env->ReleaseStringUTFChars(modelDir, model_dir_chars);
  env->ReleaseStringUTFChars(wavPath, wav_path_chars);

  LOGI("build_info=%s", asr_sdk_build_info_json());
  LOGI("model_dir=%s", model_dir.c_str());
  LOGI("wav_path=%s", wav_path.c_str());

  WavData wav;
  std::string error;
  if (!ReadWavFile(wav_path, &wav, &error)) {
    LOGE("%s", error.c_str());
    return env->NewStringUTF(("{\"error\":\"" + error + "\"}").c_str());
  }

  AsrSdkEngine* engine = nullptr;
  int status = asr_sdk_create_engine(model_dir.c_str(), &engine);
  if (status != ASR_SDK_STATUS_OK) {
    std::string result = LastErrorJson(engine, "create_engine");
    LOGE("%s", result.c_str());
    asr_sdk_destroy_engine(engine);
    return env->NewStringUTF(result.c_str());
  }

  AsrSdkStream* stream = nullptr;
  status = asr_sdk_create_stream(engine, &stream);
  if (status != ASR_SDK_STATUS_OK) {
    std::string result = LastErrorJson(engine, "create_stream");
    LOGE("%s", result.c_str());
    asr_sdk_destroy_engine(engine);
    return env->NewStringUTF(result.c_str());
  }

  status = asr_sdk_accept_pcm16(stream, wav.samples.data(),
                                static_cast<int>(wav.samples.size()),
                                wav.sample_rate);
  if (status == ASR_SDK_STATUS_OK) {
    status = asr_sdk_set_input_finished(stream);
  }
  while (status == ASR_SDK_STATUS_OK && asr_sdk_decode_ready(stream)) {
    status = asr_sdk_decode(stream);
  }
  const char* final_json =
      status == ASR_SDK_STATUS_OK ? asr_sdk_get_final_result_json(stream)
                                  : nullptr;
  std::string result =
      final_json ? final_json : LastErrorJson(stream, "decode");
  LOGI("final_result=%s", result.c_str());

  asr_sdk_destroy_stream(stream);
  asr_sdk_destroy_engine(engine);
  return env->NewStringUTF(result.c_str());
}
