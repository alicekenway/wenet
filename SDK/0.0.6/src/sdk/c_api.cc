#include "asr_sdk/c_api.h"

#include <algorithm>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "asr_sdk/asr_engine.h"
#include "asr_sdk/decode_context.h"
#include "asr_sdk/version.h"
#include "sdk/result_json.h"

struct AsrSdkEngine {
  int kind = 1;
  std::shared_ptr<asr_sdk::AsrEngine> engine;
  asr_sdk::Status last_status;
};

struct AsrSdkStream {
  int kind = 2;
  std::unique_ptr<asr_sdk::AsrStream> stream;
  asr_sdk::Status last_status;
  std::string result_json;
};

struct AsrSdkContext {
  int kind = 3;
  std::shared_ptr<asr_sdk::AsrEngine> engine;
  asr_sdk::DecodeContextConfig config;
  std::shared_ptr<const asr_sdk::DecodeContext> compiled_context;
  bool compiled = false;
  asr_sdk::Status last_status;
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
  if (kind == 3) {
    return static_cast<AsrSdkContext*>(handle)->last_status;
  }
  static thread_local asr_sdk::Status unknown =
      asr_sdk::Status::InvalidArgument("unknown handle type");
  return unknown;
}

int ContextMutationStatus(AsrSdkContext* context) {
  if (context == nullptr) {
    SetGlobalStatus(asr_sdk::Status::InvalidArgument("context is null"));
    return ToCStatus(g_last_status);
  }
  if (context->compiled) {
    context->last_status = asr_sdk::Status::FailedPrecondition(
        "context cannot be modified after compilation");
    SetGlobalStatus(context->last_status);
    return ToCStatus(context->last_status);
  }
  return ASR_SDK_STATUS_OK;
}

int SetContextError(AsrSdkContext* context, asr_sdk::Status status) {
  if (context != nullptr) {
    context->last_status = std::move(status);
    SetGlobalStatus(context->last_status);
    return ToCStatus(context->last_status);
  }
  SetGlobalStatus(std::move(status));
  return ToCStatus(g_last_status);
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
  handle->engine = std::shared_ptr<asr_sdk::AsrEngine>(
      std::move(engine_or).value());
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
  engine->last_status = asr_sdk::Status::Ok();
  SetGlobalStatus(asr_sdk::Status::Ok());
  return ASR_SDK_STATUS_OK;
}

int asr_sdk_create_context(AsrSdkEngine* engine, AsrSdkContext** out_context) {
  if (out_context == nullptr) {
    SetGlobalStatus(asr_sdk::Status::InvalidArgument("out_context is null"));
    return ToCStatus(g_last_status);
  }
  *out_context = nullptr;
  if (engine == nullptr || !engine->engine) {
    SetGlobalStatus(asr_sdk::Status::InvalidArgument("engine is null"));
    return ToCStatus(g_last_status);
  }
  auto* context = new AsrSdkContext;
  context->engine = engine->engine;
  context->last_status = asr_sdk::Status::Ok();
  *out_context = context;
  engine->last_status = asr_sdk::Status::Ok();
  SetGlobalStatus(asr_sdk::Status::Ok());
  return ASR_SDK_STATUS_OK;
}

int asr_sdk_context_add_contact_form(
    AsrSdkContext* context, const char* contact_id, const char* display_name,
    const char* spoken_form, const char* const* am_tokens, int num_am_tokens,
    int logical_word_count) {
  const int mutable_status = ContextMutationStatus(context);
  if (mutable_status != ASR_SDK_STATUS_OK) {
    return mutable_status;
  }
  if (contact_id == nullptr || display_name == nullptr || spoken_form == nullptr) {
    return SetContextError(
        context, asr_sdk::Status::InvalidArgument(
                     "contact_id, display_name, and spoken_form are required"));
  }
  if (num_am_tokens < 0 || logical_word_count < 0) {
    return SetContextError(
        context, asr_sdk::Status::InvalidArgument(
                     "num_am_tokens and logical_word_count must be non-negative"));
  }
  if (num_am_tokens > 0 && am_tokens == nullptr) {
    return SetContextError(
        context, asr_sdk::Status::InvalidArgument(
                     "am_tokens is null while num_am_tokens is positive"));
  }

  const std::string id(contact_id);
  const std::string display(display_name);
  if (id.empty() || display.empty() || std::string(spoken_form).empty()) {
    return SetContextError(
        context, asr_sdk::Status::InvalidArgument(
                     "contact_id, display_name, and spoken_form must be non-empty"));
  }

  asr_sdk::ContactSpokenForm form;
  form.text = spoken_form;
  form.logical_word_count = logical_word_count;
  form.am_tokens.reserve(static_cast<size_t>(num_am_tokens));
  for (int i = 0; i < num_am_tokens; ++i) {
    if (am_tokens[i] == nullptr) {
      return SetContextError(
          context, asr_sdk::Status::InvalidArgument("am_tokens contains null"));
    }
    form.am_tokens.emplace_back(am_tokens[i]);
  }

  auto it = std::find_if(context->config.contacts.begin(),
                         context->config.contacts.end(),
                         [&](const asr_sdk::ContactEntry& entry) {
                           return entry.contact_id == id;
                         });
  if (it == context->config.contacts.end()) {
    context->config.contacts.push_back(asr_sdk::ContactEntry{});
    it = std::prev(context->config.contacts.end());
    it->contact_id = id;
    it->display_name = display;
  } else if (it->display_name != display) {
    return SetContextError(
        context, asr_sdk::Status::InvalidArgument(
                     "repeated contact_id has a conflicting display_name"));
  }
  it->spoken_forms.push_back(std::move(form));
  context->last_status = asr_sdk::Status::Ok();
  SetGlobalStatus(asr_sdk::Status::Ok());
  return ASR_SDK_STATUS_OK;
}

int asr_sdk_context_compile(AsrSdkContext* context) {
  if (context == nullptr || !context->engine) {
    return SetContextError(
        context, asr_sdk::Status::InvalidArgument("context is null or invalid"));
  }
  if (context->compiled) {
    return SetContextError(
        context, asr_sdk::Status::FailedPrecondition(
                     "context has already been compiled"));
  }
  auto compiled_or = context->engine->CompileDecodeContext(context->config);
  if (!compiled_or.ok()) {
    return SetContextError(context, compiled_or.status());
  }
  context->compiled_context = std::move(compiled_or).value();
  context->compiled = true;
  context->last_status = asr_sdk::Status::Ok();
  SetGlobalStatus(asr_sdk::Status::Ok());
  return ASR_SDK_STATUS_OK;
}

int asr_sdk_create_stream_with_context(AsrSdkEngine* engine,
                                       const AsrSdkContext* context,
                                       AsrSdkStream** out_stream) {
  if (out_stream == nullptr) {
    SetGlobalStatus(asr_sdk::Status::InvalidArgument("out_stream is null"));
    return ToCStatus(g_last_status);
  }
  *out_stream = nullptr;
  if (engine == nullptr || !engine->engine || context == nullptr) {
    SetGlobalStatus(
        asr_sdk::Status::InvalidArgument("engine or context is null"));
    return ToCStatus(g_last_status);
  }
  if (!context->compiled || !context->compiled_context) {
    engine->last_status = asr_sdk::Status::FailedPrecondition(
        "context must be compiled before creating a stream");
    SetGlobalStatus(engine->last_status);
    return ToCStatus(engine->last_status);
  }
  if (context->engine.get() != engine->engine.get()) {
    engine->last_status = asr_sdk::Status::FailedPrecondition(
        "context was compiled by a different engine");
    SetGlobalStatus(engine->last_status);
    return ToCStatus(engine->last_status);
  }
  auto stream_or = engine->engine->CreateStream(context->compiled_context);
  if (!stream_or.ok()) {
    engine->last_status = stream_or.status();
    SetGlobalStatus(engine->last_status);
    return ToCStatus(engine->last_status);
  }
  auto* stream = new AsrSdkStream;
  stream->stream = std::move(stream_or).value();
  stream->last_status = asr_sdk::Status::Ok();
  *out_stream = stream;
  engine->last_status = asr_sdk::Status::Ok();
  SetGlobalStatus(asr_sdk::Status::Ok());
  return ASR_SDK_STATUS_OK;
}

void asr_sdk_destroy_context(AsrSdkContext* context) { delete context; }

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
