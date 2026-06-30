#include "sherpa_onnx_wenet/wenet_ctc_onnx_backend.h"

#include <algorithm>
#include <cmath>
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

bool NameIs(const std::string& name, std::initializer_list<const char*> aliases) {
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

int ParseMetadataInt(Ort::ModelMetadata* metadata,
                     Ort::AllocatorWithDefaultOptions* allocator,
                     const char* key) {
  const std::string value = ParseMetadataString(metadata, allocator, key);
  return std::atoi(value.c_str());
}

int ParseMetadataIntAny(Ort::ModelMetadata* metadata,
                        Ort::AllocatorWithDefaultOptions* allocator,
                        std::initializer_list<const char*> keys) {
  for (const char* key : keys) {
    Ort::AllocatedStringPtr value =
        metadata->LookupCustomMetadataMapAllocated(key, *allocator);
    if (value != nullptr) {
      return std::atoi(value.get());
    }
  }
  std::string message = "missing ONNX metadata: ";
  bool first = true;
  for (const char* key : keys) {
    if (!first) message += "/";
    message += key;
    first = false;
  }
  throw std::runtime_error(message);
}

std::vector<int64_t> TensorShape(const Ort::TypeInfo& info) {
  auto tensor_info = info.GetTensorTypeAndShapeInfo();
  return tensor_info.GetShape();
}

ONNXTensorElementDataType TensorType(const Ort::TypeInfo& info) {
  return info.GetTensorTypeAndShapeInfo().GetElementType();
}

float LogSumExp(const std::vector<float>& values) {
  const float max_value = *std::max_element(values.begin(), values.end());
  double sum = 0.0;
  for (float value : values) {
    sum += std::exp(static_cast<double>(value - max_value));
  }
  return max_value + static_cast<float>(std::log(sum));
}

}  // namespace

WenetCtcOnnxResource::WenetCtcOnnxResource(const std::string& model_path,
                                           int num_threads, int blank_id)
    : env_(ORT_LOGGING_LEVEL_WARNING, "wenet_ctc_onnx") {
  session_options_.SetIntraOpNumThreads(std::max(1, num_threads));
  session_options_.SetInterOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_ =
      std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
  ReadMetadata(blank_id);
  ReadNodes();
}

void WenetCtcOnnxResource::ReadMetadata(int blank_id) {
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::ModelMetadata metadata = session_->GetModelMetadata();

  const std::string model_type =
      ParseMetadataString(&metadata, &allocator, "model_type");
  if (model_type != "wenet_ctc") {
    throw std::runtime_error("expected model_type=wenet_ctc, got " +
                             model_type);
  }

  chunk_size_ = ParseMetadataInt(&metadata, &allocator, "chunk_size");
  left_chunks_ = ParseMetadataInt(&metadata, &allocator, "left_chunks");
  num_blocks_ = ParseMetadataInt(&metadata, &allocator, "num_blocks");
  head_ = ParseMetadataInt(&metadata, &allocator, "head");
  output_size_ = ParseMetadataInt(&metadata, &allocator, "output_size");
  cnn_module_kernel_ =
      ParseMetadataInt(&metadata, &allocator, "cnn_module_kernel");
  subsampling_factor_ = ParseMetadataIntAny(
      &metadata, &allocator, {"subsampling_factor", "subsampling_rate"});
  right_context_ = ParseMetadataInt(&metadata, &allocator, "right_context");
  info_.vocab_size = ParseMetadataInt(&metadata, &allocator, "vocab_size");

  if (chunk_size_ <= 0 || left_chunks_ < 0 || num_blocks_ <= 0 || head_ <= 0 ||
      output_size_ <= 0 || cnn_module_kernel_ <= 0 ||
      subsampling_factor_ <= 0 || right_context_ < 0 || info_.vocab_size <= 0) {
    throw std::runtime_error("invalid wenet_ctc ONNX metadata values");
  }
  if (output_size_ % head_ != 0) {
    throw std::runtime_error("wenet_ctc output_size must be divisible by head");
  }

  required_cache_size_ = chunk_size_ * left_chunks_;
  info_.input_window_frames =
      (chunk_size_ - 1) * subsampling_factor_ + right_context_ + 1;
  info_.input_shift_frames = chunk_size_ * subsampling_factor_;
  info_.blank_id = blank_id;
  info_.feature_dim = 80;
  info_.sample_rate = 16000;
  info_.output_frame_shift_ms = subsampling_factor_ * 10.0f;
  info_.output_is_log_probs = true;
}

std::vector<std::string> WenetCtcOnnxResource::GetNodeNames(bool input) const {
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

void WenetCtcOnnxResource::ReadNodes() {
  input_names_ = GetNodeNames(true);
  output_names_ = GetNodeNames(false);
  if (input_names_.empty() || output_names_.empty()) {
    throw std::runtime_error("ONNX model has no inputs or outputs");
  }

  const int feature_index = FindName(input_names_, {"x", "chunk"});
  const int offset_index = FindName(input_names_, {"offset"});
  const int required_cache_index =
      FindName(input_names_, {"required_cache_size"});
  const int att_cache_index = FindName(input_names_, {"attn_cache", "att_cache"});
  const int conv_cache_index = FindName(input_names_, {"conv_cache", "cnn_cache"});
  const int att_mask_index = FindName(input_names_, {"attn_mask", "att_mask"});
  if (feature_index < 0 || offset_index < 0 || required_cache_index < 0 ||
      att_cache_index < 0 || conv_cache_index < 0 || att_mask_index < 0) {
    throw std::runtime_error(
        "wenet_ctc model must expose x/chunk, offset, required_cache_size, "
        "attn_cache/att_cache, conv_cache/cnn_cache, and attn_mask/att_mask");
  }

  if (FindName(output_names_, {"log_probs"}) < 0 ||
      FindName(output_names_, {"next_att_cache", "next_attn_cache",
                               "att_cache", "attn_cache"}) < 0 ||
      FindName(output_names_, {"next_conv_cache", "next_cnn_cache",
                               "conv_cache", "cnn_cache"}) < 0) {
    throw std::runtime_error(
        "wenet_ctc model must expose log_probs, next_att_cache, and "
        "next_conv_cache outputs");
  }

  if (TensorType(session_->GetInputTypeInfo(feature_index)) !=
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("wenet_ctc feature input must be float");
  }
  const std::vector<int64_t> feature_shape =
      TensorShape(session_->GetInputTypeInfo(feature_index));
  if (feature_shape.size() != 3 ||
      (feature_shape[2] > 0 && feature_shape[2] != info_.feature_dim)) {
    throw std::runtime_error("unexpected wenet_ctc feature input shape " +
                             ShapeToString(feature_shape));
  }

  const int log_index = FindName(output_names_, {"log_probs"});
  if (TensorType(session_->GetOutputTypeInfo(log_index)) !=
      ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error("wenet_ctc log_probs output must be float");
  }
  const std::vector<int64_t> log_shape =
      TensorShape(session_->GetOutputTypeInfo(log_index));
  if (log_shape.size() != 3 ||
      (log_shape[2] > 0 && log_shape[2] != info_.vocab_size)) {
    throw std::runtime_error("unexpected wenet_ctc log_probs output shape " +
                             ShapeToString(log_shape));
  }
}

WenetCtcOnnxBackend::WenetCtcOnnxBackend(
    std::shared_ptr<WenetCtcOnnxResource> resource)
    : resource_(std::move(resource)) {
  if (!resource_) {
    throw std::runtime_error("wenet_ctc resource is null");
  }
  Reset();
}

void WenetCtcOnnxBackend::Reset() {
  const auto& r = *resource_;
  offset_ = r.required_cache_size_;
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

  const std::vector<int64_t> att_shape = {
      r.num_blocks_, r.head_, r.required_cache_size_,
      r.output_size_ / r.head_ * 2};
  att_cache_.assign(static_cast<size_t>(r.num_blocks_ * r.head_ *
                                        r.required_cache_size_ *
                                        (r.output_size_ / r.head_ * 2)),
                    0.0f);
  att_cache_value_ = Ort::Value::CreateTensor<float>(
      memory_info, att_cache_.data(), att_cache_.size(), att_shape.data(),
      att_shape.size());

  const std::vector<int64_t> conv_shape = {
      r.num_blocks_, 1, r.output_size_, r.cnn_module_kernel_ - 1};
  conv_cache_.assign(static_cast<size_t>(r.num_blocks_ * r.output_size_ *
                                         (r.cnn_module_kernel_ - 1)),
                     0.0f);
  conv_cache_value_ = Ort::Value::CreateTensor<float>(
      memory_info, conv_cache_.data(), conv_cache_.size(), conv_shape.data(),
      conv_shape.size());
}

void WenetCtcOnnxBackend::Forward(
    const float* features, int num_frames,
    std::vector<std::vector<float>>* log_probs) {
  if (features == nullptr || log_probs == nullptr) {
    throw std::runtime_error("wenet_ctc forward received null input/output");
  }
  const auto& info = resource_->Info();
  if (num_frames != info.input_window_frames) {
    throw std::runtime_error("wenet_ctc forward expected " +
                             std::to_string(info.input_window_frames) +
                             " frames, got " + std::to_string(num_frames));
  }

  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  const std::vector<int64_t> feature_shape = {
      1, info.input_window_frames, info.feature_dim};
  const size_t feature_count =
      static_cast<size_t>(info.input_window_frames * info.feature_dim);
  Ort::Value feature_tensor = Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float*>(features), feature_count,
      feature_shape.data(), feature_shape.size());

  int64_t offset_value = offset_;
  const std::vector<int64_t> scalar_shape = {1};
  Ort::Value offset_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, &offset_value, 1, scalar_shape.data(), scalar_shape.size());

  int64_t required_cache_size = resource_->required_cache_size_;
  Ort::Value required_cache_tensor = Ort::Value::CreateTensor<int64_t>(
      memory_info, &required_cache_size, 1, scalar_shape.data(),
      scalar_shape.size());

  std::vector<uint8_t> att_mask(
      static_cast<size_t>(resource_->required_cache_size_ + resource_->chunk_size_),
      1);
  if (resource_->left_chunks_ > 0) {
    const int chunk_idx =
        static_cast<int>(offset_ / resource_->chunk_size_) -
        resource_->left_chunks_;
    if (chunk_idx < resource_->left_chunks_) {
      const int unavailable =
          (resource_->left_chunks_ - chunk_idx) * resource_->chunk_size_;
      const int limit = std::min(unavailable, static_cast<int>(att_mask.size()));
      std::fill(att_mask.begin(), att_mask.begin() + limit, 0);
    }
  }
  const std::vector<int64_t> att_mask_shape = {
      1, 1, resource_->required_cache_size_ + resource_->chunk_size_};
  Ort::Value att_mask_tensor = Ort::Value::CreateTensor<bool>(
      memory_info, reinterpret_cast<bool*>(att_mask.data()), att_mask.size(),
      att_mask_shape.data(), att_mask_shape.size());

  std::vector<Ort::Value> inputs;
  inputs.reserve(resource_->input_names_.size());
  for (const std::string& name : resource_->input_names_) {
    if (NameIs(name, {"x", "chunk"})) {
      inputs.emplace_back(std::move(feature_tensor));
    } else if (NameIs(name, {"offset"})) {
      inputs.emplace_back(std::move(offset_tensor));
    } else if (NameIs(name, {"required_cache_size"})) {
      inputs.emplace_back(std::move(required_cache_tensor));
    } else if (NameIs(name, {"attn_cache", "att_cache"})) {
      inputs.emplace_back(std::move(att_cache_value_));
    } else if (NameIs(name, {"conv_cache", "cnn_cache"})) {
      inputs.emplace_back(std::move(conv_cache_value_));
    } else if (NameIs(name, {"attn_mask", "att_mask"})) {
      inputs.emplace_back(std::move(att_mask_tensor));
    } else {
      throw std::runtime_error("unsupported wenet_ctc input: " + name);
    }
  }

  const auto input_names = ToCNames(resource_->input_names_);
  const auto output_names = ToCNames(resource_->output_names_);
  std::vector<Ort::Value> outputs = resource_->session_->Run(
      Ort::RunOptions{nullptr}, input_names.data(), inputs.data(),
      inputs.size(), output_names.data(), output_names.size());
  if (outputs.size() != resource_->output_names_.size()) {
    throw std::runtime_error("unexpected wenet_ctc output count");
  }

  const int log_index = FindName(resource_->output_names_, {"log_probs"});
  const int att_index =
      FindName(resource_->output_names_, {"next_att_cache", "next_attn_cache",
                                          "att_cache", "attn_cache"});
  const int conv_index =
      FindName(resource_->output_names_, {"next_conv_cache", "next_cnn_cache",
                                          "conv_cache", "cnn_cache"});
  if (log_index < 0 || att_index < 0 || conv_index < 0) {
    throw std::runtime_error("wenet_ctc output name lookup failed");
  }

  auto logp_info = outputs[static_cast<size_t>(log_index)]
                       .GetTensorTypeAndShapeInfo();
  const std::vector<int64_t> shape = logp_info.GetShape();
  if (shape.size() != 3 || shape[0] != 1 || shape[2] != info.vocab_size) {
    throw std::runtime_error("unexpected wenet_ctc log_probs shape " +
                             ShapeToString(shape));
  }

  const int output_frames = static_cast<int>(shape[1]);
  const float* data =
      outputs[static_cast<size_t>(log_index)].GetTensorData<float>();
  log_probs->assign(static_cast<size_t>(output_frames),
                    std::vector<float>(static_cast<size_t>(info.vocab_size)));
  for (int t = 0; t < output_frames; ++t) {
    std::vector<float>& frame = (*log_probs)[static_cast<size_t>(t)];
    std::memcpy(frame.data(), data + t * info.vocab_size,
                sizeof(float) * static_cast<size_t>(info.vocab_size));
    for (float value : frame) {
      if (!std::isfinite(value)) {
        throw std::runtime_error("wenet_ctc produced non-finite log_probs");
      }
    }
    const float logsumexp = LogSumExp(frame);
    if (std::abs(logsumexp) > 1.0e-2f) {
      throw std::runtime_error("wenet_ctc output is not log-prob normalized");
    }
  }

  offset_ += output_frames;
  att_cache_value_ = std::move(outputs[static_cast<size_t>(att_index)]);
  conv_cache_value_ = std::move(outputs[static_cast<size_t>(conv_index)]);
}

std::unique_ptr<StreamingCtcBackend> WenetCtcOnnxBackend::CloneStream() const {
  return std::make_unique<WenetCtcOnnxBackend>(resource_);
}

}  // namespace asr_sdk::internal::sherpa_onnx_wenet
