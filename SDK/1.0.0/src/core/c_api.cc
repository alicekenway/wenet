#include "wenet_sdk/c_api.h"

#include <memory>
#include <string>
#include <vector>

#include "core/result_builder.h"
#include "core/stream_session.h"
#include "utils/status.h"
#include "wenet_sdk/asr_engine.h"
#include "wenet_sdk/version.h"

struct WenetSdkEngine {
  int kind = 1;
  std::unique_ptr<wenet_sdk::AsrEngine> engine;
  wenet_sdk::internal::Status last_status;
};

struct WenetSdkStream {
  int kind = 2;
  std::unique_ptr<wenet_sdk::Stream> stream;
  wenet_sdk::internal::Status last_status;
  std::string result_json;
};

namespace {

thread_local wenet_sdk::internal::Status g_last_status;

int ToCStatus(const wenet_sdk::internal::Status& status) {
  return static_cast<int>(status.code());
}

void SetGlobalStatus(wenet_sdk::internal::Status status) {
  g_last_status = std::move(status);
}

wenet_sdk::internal::Status StreamStatus(WenetSdkStream* stream) {
  if (stream == nullptr || !stream->stream) {
    return wenet_sdk::internal::Status::InvalidArgument("stream is null");
  }
  auto* session =
      dynamic_cast<wenet_sdk::internal::StreamSession*>(stream->stream.get());
  if (session == nullptr) {
    return stream->last_status;
  }
  return session->last_status();
}

}  // namespace

WenetSdkEngine* wenet_sdk_create_engine(const char* model_dir) {
  if (model_dir == nullptr) {
    SetGlobalStatus(
        wenet_sdk::internal::Status::InvalidArgument("model_dir is null"));
    return nullptr;
  }
  wenet_sdk::EngineConfig config;
  config.model_dir = model_dir;
  auto engine = wenet_sdk::AsrEngine::Create(config);
  if (!engine) {
    SetGlobalStatus(wenet_sdk::internal::Status::Internal(
        "failed to create engine; validate the model package for details"));
    return nullptr;
  }
  auto* handle = new WenetSdkEngine;
  handle->engine = std::move(engine);
  handle->last_status = wenet_sdk::internal::Status::OK();
  SetGlobalStatus(wenet_sdk::internal::Status::OK());
  return handle;
}

void wenet_sdk_destroy_engine(WenetSdkEngine* engine) {
  delete engine;
}

WenetSdkStream* wenet_sdk_create_stream(WenetSdkEngine* engine) {
  if (engine == nullptr || !engine->engine) {
    SetGlobalStatus(
        wenet_sdk::internal::Status::InvalidArgument("engine is null"));
    return nullptr;
  }
  auto stream = engine->engine->CreateStream();
  if (!stream) {
    engine->last_status = wenet_sdk::internal::Status::Internal(
        "failed to create stream");
    SetGlobalStatus(engine->last_status);
    return nullptr;
  }
  auto* handle = new WenetSdkStream;
  handle->stream = std::move(stream);
  handle->last_status = wenet_sdk::internal::Status::OK();
  return handle;
}

void wenet_sdk_destroy_stream(WenetSdkStream* stream) {
  delete stream;
}

int wenet_sdk_accept_pcm16(WenetSdkStream* stream, int sample_rate,
                           const int16_t* samples, int num_samples) {
  if (stream == nullptr || !stream->stream || num_samples < 0 ||
      (samples == nullptr && num_samples > 0)) {
    return ToCStatus(
        wenet_sdk::internal::Status::InvalidArgument("invalid stream/input"));
  }
  std::vector<float> converted(static_cast<size_t>(num_samples));
  for (int i = 0; i < num_samples; ++i) {
    converted[static_cast<size_t>(i)] =
        static_cast<float>(samples[i]) / 32768.0f;
  }
  stream->stream->AcceptWaveform(sample_rate, converted.data(),
                                 converted.size());
  stream->last_status = StreamStatus(stream);
  return ToCStatus(stream->last_status);
}

int wenet_sdk_accept_float32(WenetSdkStream* stream, int sample_rate,
                             const float* samples, int num_samples) {
  if (stream == nullptr || !stream->stream || num_samples < 0 ||
      (samples == nullptr && num_samples > 0)) {
    return ToCStatus(
        wenet_sdk::internal::Status::InvalidArgument("invalid stream/input"));
  }
  stream->stream->AcceptWaveform(sample_rate, samples,
                                 static_cast<size_t>(num_samples));
  stream->last_status = StreamStatus(stream);
  return ToCStatus(stream->last_status);
}

int wenet_sdk_decode(WenetSdkStream* stream) {
  if (stream == nullptr || !stream->stream) {
    return ToCStatus(
        wenet_sdk::internal::Status::InvalidArgument("stream is null"));
  }
  stream->stream->Decode();
  stream->last_status = StreamStatus(stream);
  return ToCStatus(stream->last_status);
}

int wenet_sdk_decode_ready(WenetSdkStream* stream) {
  if (stream == nullptr || !stream->stream) {
    return 0;
  }
  return stream->stream->DecodeReady() ? 1 : 0;
}

const char* wenet_sdk_get_result_json(WenetSdkStream* stream) {
  if (stream == nullptr || !stream->stream) {
    return nullptr;
  }
  stream->result_json =
      wenet_sdk::internal::AsrResultToJson(stream->stream->GetResult());
  return stream->result_json.c_str();
}

const char* wenet_sdk_get_final_result_json(WenetSdkStream* stream) {
  if (stream == nullptr || !stream->stream) {
    return nullptr;
  }
  stream->result_json =
      wenet_sdk::internal::AsrResultToJson(stream->stream->GetFinalResult());
  return stream->result_json.c_str();
}

void wenet_sdk_set_input_finished(WenetSdkStream* stream) {
  if (stream != nullptr && stream->stream) {
    stream->stream->SetInputFinished();
    stream->last_status = StreamStatus(stream);
  }
}

void wenet_sdk_reset_stream(WenetSdkStream* stream) {
  if (stream != nullptr && stream->stream) {
    stream->stream->Reset();
    stream->last_status = StreamStatus(stream);
  }
}

int wenet_sdk_last_error_code(void* handle) {
  if (handle == nullptr) {
    return ToCStatus(g_last_status);
  }
  const int kind = *static_cast<int*>(handle);
  if (kind == 1) {
    auto* engine = static_cast<WenetSdkEngine*>(handle);
    return ToCStatus(engine->last_status);
  }
  if (kind == 2) {
    auto* stream = static_cast<WenetSdkStream*>(handle);
    return ToCStatus(stream->last_status);
  }
  return ToCStatus(wenet_sdk::internal::Status::InvalidArgument(
      "unknown handle type"));
}

const char* wenet_sdk_last_error_message(void* handle) {
  if (handle == nullptr) {
    return g_last_status.message().c_str();
  }
  const int kind = *static_cast<int*>(handle);
  if (kind == 1) {
    auto* engine = static_cast<WenetSdkEngine*>(handle);
    return engine->last_status.message().c_str();
  }
  if (kind == 2) {
    auto* stream = static_cast<WenetSdkStream*>(handle);
    return stream->last_status.message().c_str();
  }
  static thread_local std::string unknown = "unknown handle type";
  return unknown.c_str();
}

const char* wenet_sdk_version_string(void) {
  return wenet_sdk::VersionString();
}

int wenet_sdk_abi_version(void) {
  return wenet_sdk::AbiVersion();
}
