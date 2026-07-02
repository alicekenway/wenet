#include "asr_sdk/c_api.h"

#include <memory>
#include <string>

#include "asr_sdk/asr_engine.h"
#include "asr_sdk/version.h"
#include "sdk/result_json.h"

struct AsrSdkEngine {
  int kind = 1;
  std::unique_ptr<asr_sdk::AsrEngine> engine;
  asr_sdk::Status last_status;
};

struct AsrSdkStream {
  int kind = 2;
  std::unique_ptr<asr_sdk::AsrStream> stream;
  asr_sdk::Status last_status;
  std::string result_json;
};

namespace {

thread_local asr_sdk::Status g_last_status;

int ToCStatus(const asr_sdk::Status& status) {
  return static_cast<int>(status.code());
}

void SetGlobalStatus(asr_sdk::Status status) {
  g_last_status = std::move(status);
}

const asr_sdk::Status& StatusForHandle(void* handle) {
  if (handle == nullptr) {
    return g_last_status;
  }
  const int kind = *static_cast<int*>(handle);
  if (kind == 1) {
    return static_cast<AsrSdkEngine*>(handle)->last_status;
  }
  if (kind == 2) {
    return static_cast<AsrSdkStream*>(handle)->last_status;
  }
  static thread_local asr_sdk::Status unknown =
      asr_sdk::Status::InvalidArgument("unknown handle type");
  return unknown;
}

}  // namespace

extern "C" {

int asr_sdk_create_engine(const char* model_dir, AsrSdkEngine** out_engine) {
  if (out_engine == nullptr) {
    SetGlobalStatus(asr_sdk::Status::InvalidArgument("out_engine is null"));
    return ToCStatus(g_last_status);
  }
  *out_engine = nullptr;
  if (model_dir == nullptr) {
    SetGlobalStatus(asr_sdk::Status::InvalidArgument("model_dir is null"));
    return ToCStatus(g_last_status);
  }
  asr_sdk::EngineConfig config;
  config.model_dir = model_dir;
  auto engine_or = asr_sdk::AsrEngine::Create(config);
  if (!engine_or.ok()) {
    SetGlobalStatus(engine_or.status());
    return ToCStatus(g_last_status);
  }
  auto* handle = new AsrSdkEngine;
  handle->engine = std::move(engine_or).value();
  handle->last_status = asr_sdk::Status::Ok();
  *out_engine = handle;
  SetGlobalStatus(asr_sdk::Status::Ok());
  return ASR_SDK_STATUS_OK;
}

void asr_sdk_destroy_engine(AsrSdkEngine* engine) { delete engine; }

int asr_sdk_create_stream(AsrSdkEngine* engine, AsrSdkStream** out_stream) {
  if (out_stream == nullptr) {
    SetGlobalStatus(asr_sdk::Status::InvalidArgument("out_stream is null"));
    return ToCStatus(g_last_status);
  }
  *out_stream = nullptr;
  if (engine == nullptr || !engine->engine) {
    SetGlobalStatus(asr_sdk::Status::InvalidArgument("engine is null"));
    return ToCStatus(g_last_status);
  }
  auto stream_or = engine->engine->CreateStream();
  if (!stream_or.ok()) {
    engine->last_status = stream_or.status();
    SetGlobalStatus(engine->last_status);
    return ToCStatus(engine->last_status);
  }
  auto* handle = new AsrSdkStream;
  handle->stream = std::move(stream_or).value();
  handle->last_status = asr_sdk::Status::Ok();
  *out_stream = handle;
  return ASR_SDK_STATUS_OK;
}

void asr_sdk_destroy_stream(AsrSdkStream* stream) { delete stream; }

int asr_sdk_accept_pcm16(AsrSdkStream* stream, const int16_t* samples,
                         int num_samples, int sample_rate) {
  if (stream == nullptr || !stream->stream || num_samples < 0) {
    SetGlobalStatus(asr_sdk::Status::InvalidArgument("invalid stream/input"));
    return ToCStatus(g_last_status);
  }
  stream->last_status = stream->stream->AcceptPcm16(
      samples, static_cast<size_t>(num_samples), sample_rate);
  return ToCStatus(stream->last_status);
}

int asr_sdk_decode(AsrSdkStream* stream) {
  if (stream == nullptr || !stream->stream) {
    SetGlobalStatus(asr_sdk::Status::InvalidArgument("stream is null"));
    return ToCStatus(g_last_status);
  }
  stream->last_status = stream->stream->Decode();
  return ToCStatus(stream->last_status);
}

int asr_sdk_decode_ready(AsrSdkStream* stream) {
  if (stream == nullptr || !stream->stream) {
    return 0;
  }
  return stream->stream->DecodeReady() ? 1 : 0;
}

int asr_sdk_set_input_finished(AsrSdkStream* stream) {
  if (stream == nullptr || !stream->stream) {
    SetGlobalStatus(asr_sdk::Status::InvalidArgument("stream is null"));
    return ToCStatus(g_last_status);
  }
  stream->last_status = stream->stream->SetInputFinished();
  return ToCStatus(stream->last_status);
}

int asr_sdk_reset_stream(AsrSdkStream* stream) {
  if (stream == nullptr || !stream->stream) {
    SetGlobalStatus(asr_sdk::Status::InvalidArgument("stream is null"));
    return ToCStatus(g_last_status);
  }
  stream->last_status = stream->stream->Reset();
  return ToCStatus(stream->last_status);
}

const char* asr_sdk_get_result_json(AsrSdkStream* stream) {
  if (stream == nullptr || !stream->stream) {
    SetGlobalStatus(asr_sdk::Status::InvalidArgument("stream is null"));
    return nullptr;
  }
  stream->result_json =
      asr_sdk::internal::AsrResultToJson(stream->stream->GetResult());
  return stream->result_json.c_str();
}

const char* asr_sdk_get_final_result_json(AsrSdkStream* stream) {
  if (stream == nullptr || !stream->stream) {
    SetGlobalStatus(asr_sdk::Status::InvalidArgument("stream is null"));
    return nullptr;
  }
  stream->result_json =
      asr_sdk::internal::AsrResultToJson(stream->stream->GetFinalResult());
  return stream->result_json.c_str();
}

int asr_sdk_last_error_code(void* handle) {
  return ToCStatus(StatusForHandle(handle));
}

const char* asr_sdk_last_error_message(void* handle) {
  return StatusForHandle(handle).message().c_str();
}

const char* asr_sdk_version(void) { return asr_sdk::VersionString(); }

int asr_sdk_abi_version(void) { return asr_sdk::AbiVersion(); }

const char* asr_sdk_build_info_json(void) { return asr_sdk::BuildInfoJson(); }

}  // extern "C"
