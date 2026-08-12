#include "sherpa_onnx_wenet/zipformer2_ctc_onnx_backend.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
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

int ParseMetadataInt(Ort::ModelMetadata* metadata,
                     Ort::AllocatorWithDefaultOptions* allocator,
                     const char* key) {
  Ort::AllocatedStringPtr value =
      metadata->LookupCustomMetadataMapAllocated(key, *allocator);
  if (value == nullptr) {
    throw std::runtime_error(std::string("missing ONNX metadata: ") + key);
  }
  return std::atoi(value.get());
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

TensorSpec ReadTensorSpec(const std::string& name, const Ort::TypeInfo& info) {
  auto tensor_info = info.GetTensorTypeAndShapeInfo();
  TensorSpec spec;
  spec.name = name;
  spec.type = tensor_info.GetElementType();
  spec.shape = tensor_info.GetShape();
  for (int64_t& dim : spec.shape) {
    if (dim < 0) {
      dim = 1;  // The selected plan initially supports batch size 1 only.
    }
  }
  return spec;
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

Zipformer2CtcOnnxResource::Zipformer2CtcOnnxResource(
    const std::string& model_path, int num_threads, int blank_id)
    : env_(ORT_LOGGING_LEVEL_WARNING, "zipformer2_ctc_onnx") {
  session_options_.SetIntraOpNumThreads(std::max(1, num_threads));
  session_options_.SetInterOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
  session_ =
      std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);
  ReadMetadata(blank_id);
  ReadNodes();
}

void Zipformer2CtcOnnxResource::ReadMetadata(int blank_id) {
  Ort::AllocatorWithDefaultOptions allocator;
  Ort::ModelMetadata metadata = session_->GetModelMetadata();

  const std::string model_type =
      ParseMetadataString(&metadata, &allocator, "model_type");
  if (model_type != "zipformer2") {
    throw std::runtime_error("expected model_type=zipformer2, got " +
                             model_type);
  }
  info_.input_window_frames = ParseMetadataInt(&metadata, &allocator, "T");
  info_.input_shift_frames =
      ParseMetadataInt(&metadata, &allocator, "decode_chunk_len");
  info_.blank_id = blank_id;
  info_.feature_dim = 80;
  info_.sample_rate = 16000;
}

std::vector<std::string> Zipformer2CtcOnnxResource::GetNodeNames(
    bool input) const {
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

void Zipformer2CtcOnnxResource::ReadNodes() {
  input_names_ = GetNodeNames(true);
  output_names_ = GetNodeNames(false);
  if (input_names_.empty() || output_names_.empty()) {
    throw std::runtime_error("ONNX model has no inputs or outputs");
  }
  if (input_names_[0] != "x") {
    throw std::runtime_error("Zipformer CTC input[0] must be named x");
  }
  if (output_names_[0] != "log_probs") {
    throw std::runtime_error(
        "Zipformer CTC output[0] must be named log_probs");
  }
  if (input_names_.size() != output_names_.size()) {
    throw std::runtime_error(
        "number of output states must match number of input states");
  }

  const TensorSpec feature_spec =
      ReadTensorSpec(input_names_[0], session_->GetInputTypeInfo(0));
  if (feature_spec.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      feature_spec.shape.size() != 3 || feature_spec.shape[1] !=
                                            info_.input_window_frames ||
      feature_spec.shape[2] != info_.feature_dim) {
    throw std::runtime_error("unexpected Zipformer feature input shape " +
                             ShapeToString(feature_spec.shape));
  }

  const TensorSpec log_prob_spec =
      ReadTensorSpec(output_names_[0], session_->GetOutputTypeInfo(0));
  if (log_prob_spec.type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
      log_prob_spec.shape.size() != 3 || log_prob_spec.shape[2] <= 0) {
    throw std::runtime_error("unexpected Zipformer log_probs output shape " +
                             ShapeToString(log_prob_spec.shape));
  }
  info_.vocab_size = static_cast<int>(log_prob_spec.shape[2]);

  state_input_specs_.clear();
  state_output_specs_.clear();
  for (size_t i = 1; i < input_names_.size(); ++i) {
    state_input_specs_.push_back(
        ReadTensorSpec(input_names_[i], session_->GetInputTypeInfo(i)));
    state_output_specs_.push_back(
        ReadTensorSpec(output_names_[i], session_->GetOutputTypeInfo(i)));
    if (state_input_specs_.back().type != state_output_specs_.back().type) {
      throw std::runtime_error("state type mismatch for " +
                               state_input_specs_.back().name);
    }
  }
}

Zipformer2CtcOnnxBackend::Zipformer2CtcOnnxBackend(
    std::shared_ptr<Zipformer2CtcOnnxResource> resource)
    : resource_(std::move(resource)) {
  if (!resource_) {
    throw std::runtime_error("Zipformer resource is null");
  }
  Reset();
}

std::vector<Ort::Value> Zipformer2CtcOnnxBackend::MakeZeroStates() const {
  Ort::AllocatorWithDefaultOptions allocator;
  std::vector<Ort::Value> states;
  states.reserve(resource_->StateInputSpecs().size());
  for (const TensorSpec& spec : resource_->StateInputSpecs()) {
    const int64_t count = NumElements(spec.shape);
    if (spec.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      Ort::Value value = Ort::Value::CreateTensor(
          allocator, spec.shape.data(), spec.shape.size(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
      float* data = value.GetTensorMutableData<float>();
      std::fill(data, data + count, 0.0f);
      states.emplace_back(std::move(value));
    } else if (spec.type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
      Ort::Value value = Ort::Value::CreateTensor(
          allocator, spec.shape.data(), spec.shape.size(),
          ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64);
      int64_t* data = value.GetTensorMutableData<int64_t>();
      std::fill(data, data + count, 0);
      states.emplace_back(std::move(value));
    } else {
      throw std::runtime_error("unsupported state tensor type for " +
                               spec.name);
    }
  }
  return states;
}

void Zipformer2CtcOnnxBackend::Reset() { states_ = MakeZeroStates(); }

void Zipformer2CtcOnnxBackend::Forward(
    const float* features, int num_frames,
    std::vector<std::vector<float>>* log_probs) {
  if (features == nullptr || log_probs == nullptr) {
    throw std::runtime_error("Zipformer forward received null input/output");
  }
  const auto& info = resource_->Info();
  if (num_frames != info.input_window_frames) {
    throw std::runtime_error("Zipformer forward expected " +
                             std::to_string(info.input_window_frames) +
                             " frames, got " + std::to_string(num_frames));
  }
  if (states_.size() != resource_->StateInputSpecs().size()) {
    throw std::runtime_error("Zipformer state count mismatch");
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

  std::vector<Ort::Value> inputs;
  inputs.reserve(1 + states_.size());
  inputs.emplace_back(std::move(feature_tensor));
  for (auto& state : states_) {
    inputs.emplace_back(std::move(state));
  }

  const auto input_names = ToCNames(resource_->input_names_);
  const auto output_names = ToCNames(resource_->output_names_);
  std::vector<Ort::Value> outputs = resource_->session_->Run(
      Ort::RunOptions{nullptr}, input_names.data(), inputs.data(),
      inputs.size(), output_names.data(), output_names.size());

  if (outputs.size() != resource_->output_names_.size()) {
    throw std::runtime_error("unexpected Zipformer output count");
  }
  states_.clear();
  states_.reserve(outputs.size() - 1);
  for (size_t i = 1; i < outputs.size(); ++i) {
    states_.emplace_back(std::move(outputs[i]));
  }

  auto logp_info = outputs[0].GetTensorTypeAndShapeInfo();
  const std::vector<int64_t> shape = logp_info.GetShape();
  if (shape.size() != 3 || shape[0] != 1 || shape[2] != info.vocab_size) {
    throw std::runtime_error("unexpected Zipformer log_probs shape " +
                             ShapeToString(shape));
  }
  const int output_frames = static_cast<int>(shape[1]);
  const float* data = outputs[0].GetTensorData<float>();
  log_probs->assign(static_cast<size_t>(output_frames),
                    std::vector<float>(static_cast<size_t>(info.vocab_size)));
  for (int t = 0; t < output_frames; ++t) {
    std::vector<float>& frame = (*log_probs)[static_cast<size_t>(t)];
    std::memcpy(frame.data(), data + t * info.vocab_size,
                sizeof(float) * static_cast<size_t>(info.vocab_size));
    for (float value : frame) {
      if (!std::isfinite(value)) {
        throw std::runtime_error("Zipformer produced non-finite log_probs");
      }
    }
    const float logsumexp = LogSumExp(frame);
    if (std::abs(logsumexp) > 1.0e-2f) {
      throw std::runtime_error("Zipformer output is not log-prob normalized");
    }
  }
}

std::unique_ptr<StreamingCtcBackend> Zipformer2CtcOnnxBackend::CloneStream()
    const {
  return std::make_unique<Zipformer2CtcOnnxBackend>(resource_);
}

}  // namespace asr_sdk::internal::sherpa_onnx_wenet
