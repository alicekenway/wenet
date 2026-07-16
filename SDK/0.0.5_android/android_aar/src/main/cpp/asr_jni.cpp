#include <jni.h>

#include <cstdint>
#include <cstring>
#include <string>

#include "asr_sdk/c_api.h"

namespace {

AsrSdkEngine* ToEngine(jlong handle) {
  return reinterpret_cast<AsrSdkEngine*>(static_cast<intptr_t>(handle));
}

AsrSdkStream* ToStream(jlong handle) {
  return reinterpret_cast<AsrSdkStream*>(static_cast<intptr_t>(handle));
}

jlong ToJLong(void* handle) {
  return static_cast<jlong>(reinterpret_cast<intptr_t>(handle));
}

void ThrowJava(JNIEnv* env, const char* class_name, const std::string& message) {
  jclass cls = env->FindClass(class_name);
  if (cls != nullptr) {
    env->ThrowNew(cls, message.c_str());
  }
}

void ThrowIllegalArgument(JNIEnv* env, const std::string& message) {
  ThrowJava(env, "java/lang/IllegalArgumentException", message);
}

void ThrowAsrException(JNIEnv* env, void* handle, const std::string& step) {
  const int code = asr_sdk_last_error_code(handle);
  const char* message = asr_sdk_last_error_message(handle);
  std::string full = step + " failed";
  full += " (code=" + std::to_string(code) + ")";
  if (message != nullptr && message[0] != '\0') {
    full += ": ";
    full += message;
  }
  ThrowJava(env, "com/wenet/asr/AsrException", full);
}

jstring NewUtf8String(JNIEnv* env, const char* text) {
  if (text == nullptr) {
    text = "";
  }
  const auto length = static_cast<jsize>(std::strlen(text));
  jbyteArray bytes = env->NewByteArray(length);
  if (bytes == nullptr) {
    return nullptr;
  }
  env->SetByteArrayRegion(bytes, 0, length,
                          reinterpret_cast<const jbyte*>(text));
  jstring charset = env->NewStringUTF("UTF-8");
  jclass string_class = env->FindClass("java/lang/String");
  jmethodID constructor =
      env->GetMethodID(string_class, "<init>", "([BLjava/lang/String;)V");
  auto result =
      static_cast<jstring>(env->NewObject(string_class, constructor, bytes,
                                          charset));
  env->DeleteLocalRef(bytes);
  env->DeleteLocalRef(charset);
  env->DeleteLocalRef(string_class);
  return result;
}

bool CheckStream(JNIEnv* env, jlong stream_handle) {
  if (stream_handle == 0) {
    ThrowIllegalArgument(env, "stream is closed or null");
    return false;
  }
  return true;
}

void CheckStatus(JNIEnv* env, int status, void* handle,
                 const std::string& step) {
  if (status != ASR_SDK_STATUS_OK) {
    ThrowAsrException(env, handle, step);
  }
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_wenet_asr_AsrNative_createEngine(JNIEnv* env, jclass,
                                          jstring modelDir) {
  if (modelDir == nullptr) {
    ThrowIllegalArgument(env, "modelDir is null");
    return 0;
  }
  const char* model_dir = env->GetStringUTFChars(modelDir, nullptr);
  if (model_dir == nullptr) {
    return 0;
  }
  AsrSdkEngine* engine = nullptr;
  int status = asr_sdk_create_engine(model_dir, &engine);
  env->ReleaseStringUTFChars(modelDir, model_dir);
  if (status != ASR_SDK_STATUS_OK) {
    ThrowAsrException(env, engine, "createEngine");
    asr_sdk_destroy_engine(engine);
    return 0;
  }
  return ToJLong(engine);
}

extern "C" JNIEXPORT void JNICALL
Java_com_wenet_asr_AsrNative_destroyEngine(JNIEnv*, jclass, jlong engine) {
  asr_sdk_destroy_engine(ToEngine(engine));
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_wenet_asr_AsrNative_createStream(JNIEnv* env, jclass, jlong engine) {
  if (engine == 0) {
    ThrowIllegalArgument(env, "engine is closed or null");
    return 0;
  }
  AsrSdkStream* stream = nullptr;
  int status = asr_sdk_create_stream(ToEngine(engine), &stream);
  if (status != ASR_SDK_STATUS_OK) {
    ThrowAsrException(env, ToEngine(engine), "createStream");
    asr_sdk_destroy_stream(stream);
    return 0;
  }
  return ToJLong(stream);
}

extern "C" JNIEXPORT void JNICALL
Java_com_wenet_asr_AsrNative_destroyStream(JNIEnv*, jclass, jlong stream) {
  asr_sdk_destroy_stream(ToStream(stream));
}

extern "C" JNIEXPORT void JNICALL
Java_com_wenet_asr_AsrNative_acceptPcm16(JNIEnv* env, jclass, jlong stream,
                                         jshortArray samples, jint offset,
                                         jint length, jint sampleRate) {
  if (!CheckStream(env, stream)) {
    return;
  }
  if (samples == nullptr) {
    ThrowIllegalArgument(env, "samples is null");
    return;
  }
  const jsize array_length = env->GetArrayLength(samples);
  if (offset < 0 || length < 0 || offset + length > array_length) {
    ThrowIllegalArgument(env, "invalid sample offset/length");
    return;
  }
  if (length == 0) {
    return;
  }
  jboolean is_copy = JNI_FALSE;
  jshort* data = env->GetShortArrayElements(samples, &is_copy);
  if (data == nullptr) {
    return;
  }
  int status = asr_sdk_accept_pcm16(
      ToStream(stream), reinterpret_cast<int16_t*>(data + offset), length,
      sampleRate);
  env->ReleaseShortArrayElements(samples, data, JNI_ABORT);
  CheckStatus(env, status, ToStream(stream), "acceptPcm16");
}

extern "C" JNIEXPORT jboolean JNICALL
Java_com_wenet_asr_AsrNative_decodeReady(JNIEnv* env, jclass, jlong stream) {
  if (!CheckStream(env, stream)) {
    return JNI_FALSE;
  }
  return asr_sdk_decode_ready(ToStream(stream)) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT void JNICALL
Java_com_wenet_asr_AsrNative_decode(JNIEnv* env, jclass, jlong stream) {
  if (!CheckStream(env, stream)) {
    return;
  }
  int status = asr_sdk_decode(ToStream(stream));
  CheckStatus(env, status, ToStream(stream), "decode");
}

extern "C" JNIEXPORT void JNICALL
Java_com_wenet_asr_AsrNative_setInputFinished(JNIEnv* env, jclass,
                                              jlong stream) {
  if (!CheckStream(env, stream)) {
    return;
  }
  int status = asr_sdk_set_input_finished(ToStream(stream));
  CheckStatus(env, status, ToStream(stream), "setInputFinished");
}

extern "C" JNIEXPORT void JNICALL
Java_com_wenet_asr_AsrNative_resetStream(JNIEnv* env, jclass, jlong stream) {
  if (!CheckStream(env, stream)) {
    return;
  }
  int status = asr_sdk_reset_stream(ToStream(stream));
  CheckStatus(env, status, ToStream(stream), "resetStream");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_wenet_asr_AsrNative_getResultJson(JNIEnv* env, jclass, jlong stream) {
  if (!CheckStream(env, stream)) {
    return nullptr;
  }
  const char* result = asr_sdk_get_result_json(ToStream(stream));
  if (result == nullptr) {
    ThrowAsrException(env, ToStream(stream), "getResultJson");
    return nullptr;
  }
  return NewUtf8String(env, result);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_wenet_asr_AsrNative_getFinalResultJson(JNIEnv* env, jclass,
                                                jlong stream) {
  if (!CheckStream(env, stream)) {
    return nullptr;
  }
  const char* result = asr_sdk_get_final_result_json(ToStream(stream));
  if (result == nullptr) {
    ThrowAsrException(env, ToStream(stream), "getFinalResultJson");
    return nullptr;
  }
  return NewUtf8String(env, result);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_wenet_asr_AsrNative_version(JNIEnv* env, jclass) {
  return NewUtf8String(env, asr_sdk_version());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_wenet_asr_AsrNative_abiVersion(JNIEnv*, jclass) {
  return asr_sdk_abi_version();
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_wenet_asr_AsrNative_buildInfoJson(JNIEnv* env, jclass) {
  return NewUtf8String(env, asr_sdk_build_info_json());
}
