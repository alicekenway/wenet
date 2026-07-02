#include "sherpa_onnx_wenet/nemo_ctc_onnx_backend.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace asr_sdk::internal::sherpa_onnx_wenet {
namespace {

std::string ShapeToString(const std::vector<int64_t>& shape) {
  std::ostringstream os;
  os << "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) os << ",";
    os << shape[i];
  }
  os << "]";
  return os.str();
}

std::vector<const char*> ToCNames(const std::vector<std::string>& names) {
  std::vector<const char*> out;
  out.reserve(names.size());
  for (const auto& name : names) {
    out.push_back(name.c_str());
  }
  return out;
}

bool NameIs(const std::string& name,
            std::initializer_list<const char*> aliases) {
  for (const char* alias : aliases) {
    if (name == alias) {
      return true;
    }
  }
  return false;
}

int FindName(const std::vector<std::string>& names,
             std::initializer_list<const char*> aliases) {
  for (size_t i = 0; i < names.size(); ++i) {
    if (NameIs(names[i], aliases)) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

std::string ParseMetadataString(Ort::ModelMetadata* metadata,
                                Ort::AllocatorWithDefaultOptions* allocator,
                                const char* key) {
  Ort::AllocatedStringPtr value =
      metadata->LookupCustomMetadataMapAllocated(key, *allocator);
  if (value == nullptr) {
    throw std::runtime_error(std::string("missing ONNX metadata: ") + key);
  }
  return value.get();
}

std::string ParseMetadataStringOr(
    Ort::ModelMetadata* metadata, Ort::AllocatorWithDefaultOptions* allocator,
    const char* key, const std::string& fallback) {
  Ort::AllocatedStringPtr value =
      metadata->LookupCustomMetadataMapAllocated(key, *allocator);
  if (value == nullptr) {
    return fallback;
  }
  return value.get();
}

int ParseMetadataInt(Ort::ModelMetadata* metadata,
                     Ort::AllocatorWithDefaultOptions* allocator,
                     const char* key) {
  const std::string value = ParseMetadataString(metadata, allocator, key);
  return std::atoi(value.c_str());
}

int ParseMetadataIntOr(Ort::ModelMetadata* metadata,
                       Ort::AllocatorWithDefaultOptions* allocator,
                       const char* key, int fallback) {
  Ort::AllocatedStringPtr value =
      metadata->LookupCustomMetadataMapAllocated(key, *allocator);
  if (value == nullptr) {
    return fallback;
  }
  return std::atoi(value.get());
}

std::vector<int64_t> TensorShape(const Ort::TypeInfo& info) {
  return info.GetTensorTypeAndShapeInfo().GetShape();
}

ONNXTensorElementDataType TensorType(const Ort::TypeInfo& info) {
  return info.GetTensorTypeAndShapeInfo().GetElementType();
}

int64_t NumElements(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t dim : shape) {
    if (dim <= 0) {
      throw std::runtime_error("invalid tensor shape " + ShapeToString(shape));
    }
    n *= dim;
  }
  return n;
}

std::vector<int64_t> StaticShapeFromMetadata(Ort::ModelMetadata* metadata,
                                             Ort::AllocatorWithDefaultOptions* allocator,
                                             const char* prefix) {
  const std::string dim1_key = std::string(prefix) + "_dim1";
  const std::string dim2_key = std::string(prefix) + "_dim2";
  const std::string dim3_key = std::string(prefix) + "_dim3";
  return {1,
          ParseMetadataInt(metadata, allocator, dim1_key.c_str()),
          ParseMetadataInt(metadata, allocator, dim2_key.c_str()),
          ParseMetadataInt(metadata, allocator, dim3_key.c_str())};
}

float LogSumExp(const std::vector<float>& values) {
  const float max_value = *std::max_element(values.begin(), values.end());
  double sum = 0.0;
  for (float value : values) {
    sum += std::exp(static_cast<double>(value - max_value));
  }
  return max_value + static_cast<float>(std::log(sum));
}

void CopyFloatTensor(Ort::Value* value, std::vector<float>* out) {
  const auto info = value->GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("expected float tensor output");
  }
  const std::vector<int64_t> shape = info.GetShape();
  const int64_t count = NumElements(shape);
  const float* data = value->GetTensorData<float>();
  out->assign(data, data + count);
}

void CopyInt64Tensor(Ort::Value* value, std::vector<int64_t>* out) {
  const auto info = value->GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    throw std::runtime_error("expected int64 tensor output");
  }
  const std::vector<int64_t> shape = info.GetShape();
  const int64_t count = NumElements(shape);
  const int64_t* data = value->GetTensorData<int64_t>();
  out->assign(data, data + count);
}

}  // namespace

NemoCtcOnnxResource::NemoCtcOnnxResource(const std::string& model_path,
                                         int num_threads, int blank_id)
    : env_(ORT_LOGGING_LEVEL_WARNING, "nemo_ctc_onnx") {
  session_options_.SetIntraOpNumThreads(std::max(1, num_threads));
  session_options_.SetInterOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_ =
      std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
  ReadMetadata(blank_id);
  ReadNodes();
}

void NemoCtcOnnxResource::ReadMetadata(int blank_id) {
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::ModelMetadata metadata = session_->GetModelMetadata();

  const std::string model_type =
      ParseMetadataString(&metadata, &allocator, "model_type");
  const std::string model_author =
      ParseMetadataStringOr(&metadata, &allocator, "model_author", "");
  if (model_type != "EncDecHybridRNNTCTCBPEModel" &&
      model_type != "nemo_ctc" && model_type != "nemo_asr_onnx" &&
      model_author != "NeMo") {
    throw std::runtime_error("expected a NeMo CTC ONNX model, got " +
                             model_type);
  }

  info_.input_window_frames =
      ParseMetadataIntOr(&metadata, &allocator, "window_size", 25);
  info_.input_shift_frames =
      ParseMetadataIntOr(&metadata, &allocator, "chunk_shift", 16);
  if (info_.input_window_frames <= 0 || info_.input_shift_frames <= 0 ||
      info_.input_window_frames < info_.input_shift_frames) {
    throw std::runtime_error("invalid NeMo window_size/chunk_shift metadata");
  }
  info_.blank_id = blank_id;
  info_.feature_dim = 80;
  info_.sample_rate = 16000;
  const int subsampling_factor =
      ParseMetadataIntOr(&metadata, &allocator, "subsampling_factor", 8);
  info_.output_frame_shift_ms = subsampling_factor * 10.0f;
  info_.output_is_log_probs = true;
  cache_last_channel_shape_ =
      StaticShapeFromMetadata(&metadata, &allocator, "cache_last_channel");
  cache_last_time_shape_ =
      StaticShapeFromMetadata(&metadata, &allocator, "cache_last_time");
}

std::vector<std::string> NemoCtcOnnxResource::GetNodeNames(bool input) const {
  Ort::AllocatorWithDefaultOptions allocator;
  const size_t count = input ? session_->GetInputCount()
                             : session_->GetOutputCount();
  std::vector<std::string> names;
  names.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    Ort::AllocatedStringPtr ptr =
        input ? session_->GetInputNameAllocated(i, allocator)
              : session_->GetOutputNameAllocated(i, allocator);
    names.emplace_back(ptr.get());
  }
  return names;
}

void NemoCtcOnnxResource::ReadNodes() {
  input_names_ = GetNodeNames(true);
  output_names_ = GetNodeNames(false);
  if (input_names_.empty() || output_names_.empty()) {
    throw std::runtime_error("ONNX model has no inputs or outputs");
  }

  const int audio_index = FindName(input_names_, {"audio_signal"});
  const int length_index = FindName(input_names_, {"length"});
  const int cache_channel_index =
      FindName(input_names_, {"cache_last_channel"});
  const int cache_time_index = FindName(input_names_, {"cache_last_time"});
  const int cache_len_index =
      FindName(input_names_, {"cache_last_channel_len"});
  if (audio_index < 0 || length_index < 0 || cache_channel_index < 0 ||
      cache_time_index < 0 || cache_len_index < 0) {
    throw std::runtime_error(
        "NeMo model must expose audio_signal, length, cache_last_channel, "
        "cache_last_time, and cache_last_channel_len inputs");
  }

  const int logprob_index = FindName(output_names_, {"logprobs", "log_probs"});
  const int encoded_lengths_index = FindName(output_names_, {"encoded_lengths"});
  const int next_channel_index =
      FindName(output_names_, {"cache_last_channel_next"});
  const int next_time_index = FindName(output_names_, {"cache_last_time_next"});
  const int next_len_index =
      FindName(output_names_, {"cache_last_channel_next_len"});
  if (logprob_index < 0 || encoded_lengths_index < 0 ||
      next_channel_index < 0 || next_time_index < 0 || next_len_index < 0) {
    throw std::runtime_error(
        "NeMo model must expose logprobs, encoded_lengths, "
        "cache_last_channel_next, cache_last_time_next, and "
        "cache_last_channel_next_len outputs");
  }

  if (TensorType(session_->GetInputTypeInfo(audio_index)) !=
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("NeMo audio_signal input must be float");
  }
  const std::vector<int64_t> audio_shape =
      TensorShape(session_->GetInputTypeInfo(audio_index));
  if (audio_shape.size() != 3 ||
      (audio_shape[1] > 0 && audio_shape[1] != info_.feature_dim)) {
    throw std::runtime_error("unexpected NeMo audio_signal shape " +
                             ShapeToString(audio_shape));
  }

  if (TensorType(session_->GetOutputTypeInfo(logprob_index)) !=
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("NeMo logprobs output must be float");
  }
  const std::vector<int64_t> logprob_shape =
      TensorShape(session_->GetOutputTypeInfo(logprob_index));
  if (logprob_shape.size() != 3 || logprob_shape[2] <= 0) {
    throw std::runtime_error("unexpected NeMo logprobs shape " +
                             ShapeToString(logprob_shape));
  }
  info_.vocab_size = static_cast<int>(logprob_shape[2]);
}

NemoCtcOnnxBackend::NemoCtcOnnxBackend(
    std::shared_ptr<NemoCtcOnnxResource> resource)
    : resource_(std::move(resource)) {
  if (!resource_) {
    throw std::runtime_error("NeMo CTC resource is null");
  }
  Reset();
}

void NemoCtcOnnxBackend::Reset() {
  cache_last_channel_.assign(
      static_cast<size_t>(NumElements(resource_->cache_last_channel_shape_)),
      0.0f);
  cache_last_time_.assign(
      static_cast<size_t>(NumElements(resource_->cache_last_time_shape_)),
      0.0f);
  cache_last_channel_len_.assign(1, 0);
}

void NemoCtcOnnxBackend::Forward(
    const float* features, int num_frames,
    std::vector<std::vector<float>>* log_probs) {
  if (features == nullptr || log_probs == nullptr) {
    throw std::runtime_error("NeMo CTC forward received null input/output");
  }
  const auto& info = resource_->Info();
  if (num_frames != info.input_window_frames) {
    throw std::runtime_error("NeMo CTC forward expected " +
                             std::to_string(info.input_window_frames) +
                             " frames, got " + std::to_string(num_frames));
  }

  std::vector<float> audio_signal(
      static_cast<size_t>(info.feature_dim * num_frames));
  for (int t = 0; t < num_frames; ++t) {
    for (int d = 0; d < info.feature_dim; ++d) {
      audio_signal[static_cast<size_t>(d * num_frames + t)] =
          features[static_cast<size_t>(t * info.feature_dim + d)];
    }
  }
  std::vector<int64_t> length = {num_frames};

  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  const std::vector<int64_t> audio_shape = {1, info.feature_dim, num_frames};
  const std::vector<int64_t> length_shape = {1};

  Ort::Value audio_tensor = Ort::Value::CreateTensor<float>(
      memory_info, audio_signal.data(), audio_signal.size(),
      audio_shape.data(), audio_shape.size());
  Ort::Value length_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, length.data(), length.size(), length_shape.data(),
      length_shape.size());
  Ort::Value channel_tensor = Ort::Value::CreateTensor<float>(
      memory_info, cache_last_channel_.data(), cache_last_channel_.size(),
      resource_->cache_last_channel_shape_.data(),
      resource_->cache_last_channel_shape_.size());
  Ort::Value time_tensor = Ort::Value::CreateTensor<float>(
      memory_info, cache_last_time_.data(), cache_last_time_.size(),
      resource_->cache_last_time_shape_.data(),
      resource_->cache_last_time_shape_.size());
  Ort::Value cache_len_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, cache_last_channel_len_.data(),
      cache_last_channel_len_.size(), length_shape.data(), length_shape.size());

  std::vector<Ort::Value> inputs;
  inputs.reserve(5);
  inputs.emplace_back(std::move(audio_tensor));
  inputs.emplace_back(std::move(length_tensor));
  inputs.emplace_back(std::move(channel_tensor));
  inputs.emplace_back(std::move(time_tensor));
  inputs.emplace_back(std::move(cache_len_tensor));

  const std::vector<std::string> input_names = {
      "audio_signal", "length", "cache_last_channel", "cache_last_time",
      "cache_last_channel_len"};
  const std::vector<std::string> output_names = {
      "logprobs", "encoded_lengths", "cache_last_channel_next",
      "cache_last_time_next", "cache_last_channel_next_len"};
  const auto input_c_names = ToCNames(input_names);
  const auto output_c_names = ToCNames(output_names);
  std::vector<Ort::Value> outputs = resource_->session_->Run(
      Ort::RunOptions{nullptr}, input_c_names.data(), inputs.data(),
      inputs.size(), output_c_names.data(), output_c_names.size());

  if (outputs.size() != output_names.size()) {
    throw std::runtime_error("unexpected NeMo CTC output count");
  }

  auto logprob_info = outputs[0].GetTensorTypeAndShapeInfo();
  const std::vector<int64_t> shape = logprob_info.GetShape();
  if (shape.size() != 3 || shape[0] != 1 || shape[2] != info.vocab_size) {
    throw std::runtime_error("unexpected NeMo logprobs output shape " +
                             ShapeToString(shape));
  }
  int output_frames = static_cast<int>(shape[1]);
  const int64_t* encoded_lengths = outputs[1].GetTensorData<int64_t>();
  if (encoded_lengths[0] > 0 && encoded_lengths[0] < output_frames) {
    output_frames = static_cast<int>(encoded_lengths[0]);
  }
  const float* data = outputs[0].GetTensorData<float>();
  log_probs->assign(static_cast<size_t>(output_frames),
                    std::vector<float>(static_cast<size_t>(info.vocab_size)));
  for (int t = 0; t < output_frames; ++t) {
    std::vector<float>& frame = (*log_probs)[static_cast<size_t>(t)];
    std::memcpy(frame.data(), data + t * info.vocab_size,
                sizeof(float) * static_cast<size_t>(info.vocab_size));
    for (float value : frame) {
      if (!std::isfinite(value)) {
        throw std::runtime_error("NeMo CTC produced non-finite logprobs");
      }
    }
    const float logsumexp = LogSumExp(frame);
    if (std::abs(logsumexp) > 1.0e-1f) {
      throw std::runtime_error("NeMo CTC output is not log-prob normalized");
    }
  }

  CopyFloatTensor(&outputs[2], &cache_last_channel_);
  CopyFloatTensor(&outputs[3], &cache_last_time_);
  CopyInt64Tensor(&outputs[4], &cache_last_channel_len_);
  if (cache_last_channel_len_.empty()) {
    throw std::runtime_error("NeMo CTC cache length output is empty");
  }
}

std::unique_ptr<StreamingCtcBackend> NemoCtcOnnxBackend::CloneStream() const {
  return std::make_unique<NemoCtcOnnxBackend>(resource_);
}

}  // namespace asr_sdk::internal::sherpa_onnx_wenet
