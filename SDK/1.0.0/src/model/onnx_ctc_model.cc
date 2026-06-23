#include "model/onnx_ctc_model.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <utility>

#include "model/tensor_utils.h"

#ifdef WENETSDK_ENABLE_ONNX
#include "onnxruntime_cxx_api.h"  // NOLINT
#endif

namespace wenet_sdk::internal {

namespace {

float LogProb(float probability) {
  return std::log(std::max(probability, 1.0e-20f));
}

#ifndef WENETSDK_ENABLE_ONNX
float MeanAbsEnergy(const std::vector<std::vector<float>>& features,
                    size_t begin, size_t end) {
  double sum = 0.0;
  int count = 0;
  for (size_t i = begin; i < end; ++i) {
    for (float v : features[i]) {
      sum += std::abs(v);
      ++count;
    }
  }
  return count > 0 ? static_cast<float>(sum / count) : 0.0f;
}
#endif

#ifdef WENETSDK_ENABLE_ONNX
std::vector<float> ConvertOutputFrame(const float* data, int vocab_size,
                                      const std::string& output_type) {
  std::vector<float> frame(data, data + vocab_size);
  if (output_type == "logit" || output_type == "logits") {
    return LogSoftmax(frame);
  }
  if (output_type == "prob" || output_type == "probability") {
    for (float& value : frame) {
      value = LogProb(value);
    }
  }
  return frame;
}

std::vector<std::string> GetNodeNames(Ort::Session* session, bool inputs) {
  Ort::AllocatorWithDefaultOptions allocator;
  const size_t count =
      inputs ? session->GetInputCount() : session->GetOutputCount();
  std::vector<std::string> names;
  names.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    Ort::AllocatedStringPtr ptr =
        inputs ? session->GetInputNameAllocated(i, allocator)
               : session->GetOutputNameAllocated(i, allocator);
    names.emplace_back(ptr.get());
  }
  return names;
}

bool HasName(const std::vector<std::string>& names, const std::string& name) {
  return std::find(names.begin(), names.end(), name) != names.end();
}

std::vector<const char*> ToCNames(const std::vector<std::string>& names) {
  std::vector<const char*> c_names;
  c_names.reserve(names.size());
  for (const auto& name : names) {
    c_names.push_back(name.c_str());
  }
  return c_names;
}

int ReadMetadataInt(Ort::Session* session, const char* key, int fallback) {
  try {
    Ort::AllocatorWithDefaultOptions allocator;
    auto metadata = session->GetModelMetadata();
    Ort::AllocatedStringPtr value =
        metadata.LookupCustomMetadataMapAllocated(key, allocator);
    if (value == nullptr) {
      return fallback;
    }
    return std::atoi(value.get());
  } catch (...) {
    return fallback;
  }
}

Status CopyOrtOutput(Ort::Value* value, int vocab_size,
                     const std::string& output_type,
                     std::vector<std::vector<float>>* log_probs) {
  auto info = value->GetTensorTypeAndShapeInfo();
  const auto shape = info.GetShape();
  if (shape.size() != 2 && shape.size() != 3) {
    return Status::InvalidArgument("ONNX CTC output must have rank 2 or 3");
  }

  int rows = 0;
  int cols = 0;
  if (shape.size() == 3) {
    rows = static_cast<int>(shape[1]);
    cols = static_cast<int>(shape[2]);
  } else {
    rows = static_cast<int>(shape[0]);
    cols = static_cast<int>(shape[1]);
  }
  if (rows < 0 || cols != vocab_size) {
    return Status::InvalidArgument(
        "ONNX output vocab dimension does not match tokens.txt");
  }

  const float* data = value->GetTensorData<float>();
  log_probs->clear();
  log_probs->reserve(static_cast<size_t>(rows));
  for (int r = 0; r < rows; ++r) {
    log_probs->push_back(
        ConvertOutputFrame(data + r * cols, vocab_size, output_type));
  }
  return Status::OK();
}
#endif

}  // namespace

struct OnnxCtcModel::OrtState {
#ifdef WENETSDK_ENABLE_ONNX
  static Ort::Env& Env() {
    static Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "wenet_lite_sdk");
    return env;
  }

  Ort::SessionOptions session_options;
  std::unique_ptr<Ort::Session> encoder_session;
  std::unique_ptr<Ort::Session> ctc_session;
  std::vector<std::string> encoder_in_names;
  std::vector<std::string> encoder_out_names;
  std::vector<std::string> ctc_in_names;
  std::vector<std::string> ctc_out_names;

  Ort::Value att_cache{nullptr};
  Ort::Value cnn_cache{nullptr};
  std::vector<float> att_cache_data;
  std::vector<float> cnn_cache_data;

  int offset = 0;
  int encoder_output_size = 0;
  int num_blocks = 0;
  int head = 0;
  int cnn_module_kernel = 0;
  int chunk_size = 0;
  int num_left_chunks = 0;
#endif
};

OnnxCtcModel::OnnxCtcModel(ModelMetadata metadata, int vocab_size)
    : metadata_(std::move(metadata)), vocab_size_(vocab_size) {}

OnnxCtcModel::~OnnxCtcModel() = default;

Status OnnxCtcModel::Init() {
  if (vocab_size_ <= 0) {
    return Status::InvalidArgument("vocab size must be positive");
  }
  if (metadata_.feature_dim <= 0) {
    return Status::InvalidArgument("feature_dim must be positive");
  }
  auto status = ValidateModelPackageFiles(metadata_, true);
  if (!status.ok()) {
    return status;
  }
#ifdef WENETSDK_ENABLE_ONNX
  return InitOrtBackend();
#else
  return Status::OK();
#endif
}

void OnnxCtcModel::Reset() {
  chunk_index_ = 0;
#ifdef WENETSDK_ENABLE_ONNX
  if (ort_ == nullptr) {
    return;
  }
  ort_->offset = 0;
  if (ort_->num_blocks <= 0 || ort_->head <= 0 ||
      ort_->encoder_output_size <= 0 || ort_->cnn_module_kernel <= 0) {
    return;
  }
  Ort::MemoryInfo memory_info =
      Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
  const int required_cache_size = ort_->chunk_size * ort_->num_left_chunks;
  ort_->offset = required_cache_size;

  const int att_cache_size =
      ort_->num_blocks * ort_->head * required_cache_size *
      (ort_->encoder_output_size / ort_->head) * 2;
  ort_->att_cache_data.assign(static_cast<size_t>(std::max(0, att_cache_size)),
                              0.0f);
  const int64_t att_shape[] = {
      ort_->num_blocks, ort_->head, required_cache_size,
      (ort_->encoder_output_size / ort_->head) * 2};
  ort_->att_cache = Ort::Value::CreateTensor<float>(
      memory_info, ort_->att_cache_data.data(), ort_->att_cache_data.size(),
      att_shape, 4);

  const int cnn_cache_size = ort_->num_blocks * ort_->encoder_output_size *
                             (ort_->cnn_module_kernel - 1);
  ort_->cnn_cache_data.assign(static_cast<size_t>(std::max(0, cnn_cache_size)),
                              0.0f);
  const int64_t cnn_shape[] = {ort_->num_blocks, 1, ort_->encoder_output_size,
                               ort_->cnn_module_kernel - 1};
  ort_->cnn_cache = Ort::Value::CreateTensor<float>(
      memory_info, ort_->cnn_cache_data.data(), ort_->cnn_cache_data.size(),
      cnn_shape, 4);
#endif
}

Status OnnxCtcModel::ForwardChunk(
    const std::vector<std::vector<float>>& features,
    std::vector<std::vector<float>>* log_probs) {
  if (log_probs == nullptr) {
    return Status::InvalidArgument("log_probs output is null");
  }
  log_probs->clear();
  if (features.empty()) {
    return Status::OK();
  }
  for (const auto& frame : features) {
    if (static_cast<int>(frame.size()) != metadata_.feature_dim) {
      return Status::InvalidArgument("feature dimension mismatch");
    }
  }

  const int stride = std::max(1, metadata_.subsampling_rate);
  const int output_frames =
      std::max(1, static_cast<int>((features.size() + stride - 1) / stride));
  log_probs->reserve(static_cast<size_t>(output_frames));

#ifdef WENETSDK_ENABLE_ONNX
  return ForwardOrtChunk(features, log_probs);
#else
  for (int t = 0; t < output_frames; ++t) {
    const size_t begin = std::min(features.size(), static_cast<size_t>(t * stride));
    const size_t end =
        std::min(features.size(), begin + static_cast<size_t>(stride));
    const float energy = MeanAbsEnergy(features, begin, end);

    std::vector<float> frame(static_cast<size_t>(vocab_size_),
                             LogProb(0.0001f));
    frame[static_cast<size_t>(metadata_.vocab.blank_id)] = LogProb(0.98f);
    if (vocab_size_ > 1 && energy > 1.0f) {
      const int id = 1 + ((chunk_index_ + t) % (vocab_size_ - 1));
      frame[static_cast<size_t>(metadata_.vocab.blank_id)] = LogProb(0.05f);
      frame[static_cast<size_t>(id)] = LogProb(0.90f);
    }
    log_probs->push_back(std::move(frame));
  }
  ++chunk_index_;
  return Status::OK();
#endif
}

const char* OnnxCtcModel::BackendName() const {
#ifdef WENETSDK_ENABLE_ONNX
  return "onnxruntime";
#else
  return "deterministic-onnx-stub";
#endif
}

Status OnnxCtcModel::InitOrtBackend() {
#ifdef WENETSDK_ENABLE_ONNX
  try {
    ort_ = std::make_unique<OrtState>();
    ort_->session_options.SetIntraOpNumThreads(
        std::max(1, metadata_.runtime.intra_op_num_threads));
    ort_->session_options.SetInterOpNumThreads(
        std::max(1, metadata_.runtime.inter_op_num_threads));
    ort_->session_options.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    const auto encoder_path = metadata_.Resolve(metadata_.onnx.encoder);
    ort_->encoder_session = std::make_unique<Ort::Session>(
        OrtState::Env(), encoder_path.c_str(), ort_->session_options);

    const auto ctc_path = metadata_.Resolve(metadata_.onnx.ctc);
    if (!metadata_.onnx.ctc.empty() && std::filesystem::exists(ctc_path) &&
        std::filesystem::file_size(ctc_path) > 0) {
      ort_->ctc_session = std::make_unique<Ort::Session>(
          OrtState::Env(), ctc_path.c_str(), ort_->session_options);
    }

    ort_->encoder_in_names = GetNodeNames(ort_->encoder_session.get(), true);
    ort_->encoder_out_names = GetNodeNames(ort_->encoder_session.get(), false);
    if (ort_->ctc_session) {
      ort_->ctc_in_names = GetNodeNames(ort_->ctc_session.get(), true);
      ort_->ctc_out_names = GetNodeNames(ort_->ctc_session.get(), false);
    }

    ort_->encoder_output_size =
        ReadMetadataInt(ort_->encoder_session.get(), "output_size", 0);
    ort_->num_blocks =
        ReadMetadataInt(ort_->encoder_session.get(), "num_blocks", 0);
    ort_->head = ReadMetadataInt(ort_->encoder_session.get(), "head", 0);
    ort_->cnn_module_kernel =
        ReadMetadataInt(ort_->encoder_session.get(), "cnn_module_kernel", 0);
    ort_->chunk_size = ReadMetadataInt(
        ort_->encoder_session.get(), "chunk_size", metadata_.streaming.chunk_size);
    ort_->num_left_chunks =
        ReadMetadataInt(ort_->encoder_session.get(), "left_chunks",
                        metadata_.streaming.num_left_chunks);

    Reset();
    return Status::OK();
  } catch (const std::exception& e) {
    return Status::Internal(std::string("failed to initialize ONNX Runtime: ") +
                            e.what());
  }
#else
  return Status::Unavailable("ONNX Runtime support is not compiled in");
#endif
}

Status OnnxCtcModel::ForwardOrtChunk(
    const std::vector<std::vector<float>>& features,
    std::vector<std::vector<float>>* log_probs) {
#ifdef WENETSDK_ENABLE_ONNX
  if (ort_ == nullptr || !ort_->encoder_session) {
    return Status::Internal("ONNX Runtime session is not initialized");
  }
  try {
    // Conv2d subsampling in WeNet ONNX models cannot process a 1-2 frame tail.
    // Those frames do not have enough context to produce a valid encoder frame.
    if (features.size() < 3) {
      log_probs->clear();
      return Status::OK();
    }

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    std::vector<float> feats;
    feats.reserve(features.size() * static_cast<size_t>(metadata_.feature_dim));
    for (const auto& frame : features) {
      feats.insert(feats.end(), frame.begin(), frame.end());
    }
    const int64_t feats_shape[] = {
        1, static_cast<int64_t>(features.size()), metadata_.feature_dim};
    Ort::Value feats_ort = Ort::Value::CreateTensor<float>(
        memory_info, feats.data(), feats.size(), feats_shape, 3);

    const bool has_cache =
        HasName(ort_->encoder_in_names, metadata_.onnx.att_cache_input_name) ||
        HasName(ort_->encoder_in_names, "att_cache");

    std::vector<Ort::Value> encoder_inputs;
    std::vector<int64_t> offset_value{ort_->offset};
    std::vector<int64_t> required_cache_size{
        ort_->chunk_size * ort_->num_left_chunks};
    std::vector<uint8_t> att_mask;
    Ort::Value offset_ort{nullptr};
    Ort::Value required_cache_size_ort{nullptr};
    Ort::Value att_mask_ort{nullptr};

    for (const auto& name : ort_->encoder_in_names) {
      if (name == metadata_.onnx.chunk_input_name || name == "chunk" ||
          (encoder_inputs.empty() && !has_cache)) {
        encoder_inputs.emplace_back(std::move(feats_ort));
      } else if (name == metadata_.onnx.offset_input_name || name == "offset") {
        offset_ort = Ort::Value::CreateTensor<int64_t>(
            memory_info, offset_value.data(), offset_value.size(),
            std::vector<int64_t>{}.data(), 0);
        encoder_inputs.emplace_back(std::move(offset_ort));
      } else if (name == "required_cache_size") {
        required_cache_size_ort = Ort::Value::CreateTensor<int64_t>(
            memory_info, required_cache_size.data(), required_cache_size.size(),
            std::vector<int64_t>{}.data(), 0);
        encoder_inputs.emplace_back(std::move(required_cache_size_ort));
      } else if (name == metadata_.onnx.att_cache_input_name ||
                 name == "att_cache") {
        encoder_inputs.emplace_back(std::move(ort_->att_cache));
      } else if (name == metadata_.onnx.cnn_cache_input_name ||
                 name == "cnn_cache") {
        encoder_inputs.emplace_back(std::move(ort_->cnn_cache));
      } else if (name == "att_mask") {
        const int model_chunk_size =
            ort_->chunk_size > 0 ? ort_->chunk_size
                                 : metadata_.streaming.chunk_size;
        const int mask_size =
            required_cache_size[0] + model_chunk_size;
        att_mask.assign(static_cast<size_t>(std::max(1, mask_size)), 0);
        const int valid_size = std::min(
            mask_size, model_chunk_size * static_cast<int>(chunk_index_ + 1));
        std::fill(att_mask.end() - std::max(0, valid_size), att_mask.end(), 1);
        const int64_t mask_shape[] = {1, 1, static_cast<int64_t>(att_mask.size())};
        att_mask_ort = Ort::Value::CreateTensor<bool>(
            memory_info, reinterpret_cast<bool*>(att_mask.data()),
            att_mask.size(), mask_shape, 3);
        encoder_inputs.emplace_back(std::move(att_mask_ort));
      } else {
        return Status::Unavailable("unsupported ONNX encoder input: " + name);
      }
    }

    const auto encoder_in = ToCNames(ort_->encoder_in_names);
    const auto encoder_out = ToCNames(ort_->encoder_out_names);
    std::vector<Ort::Value> encoder_outputs = ort_->encoder_session->Run(
        Ort::RunOptions{nullptr}, encoder_in.data(), encoder_inputs.data(),
        encoder_inputs.size(), encoder_out.data(), encoder_out.size());

    Ort::Value* ctc_output = nullptr;
    std::vector<Ort::Value> ctc_outputs;
    if (ort_->ctc_session) {
      std::vector<Ort::Value> ctc_inputs;
      ctc_inputs.emplace_back(std::move(encoder_outputs[0]));
      const auto ctc_in = ToCNames(ort_->ctc_in_names);
      const auto ctc_out = ToCNames(ort_->ctc_out_names);
      ctc_outputs = ort_->ctc_session->Run(
          Ort::RunOptions{nullptr}, ctc_in.data(), ctc_inputs.data(),
          ctc_inputs.size(), ctc_out.data(), ctc_out.size());
      ctc_output = &ctc_outputs[0];
    } else {
      ctc_output = &encoder_outputs[0];
    }

    if (encoder_outputs.size() >= 3 && has_cache) {
      ort_->att_cache = std::move(encoder_outputs[1]);
      ort_->cnn_cache = std::move(encoder_outputs[2]);
      auto shape = ctc_output->GetTensorTypeAndShapeInfo().GetShape();
      if (shape.size() >= 2) {
        ort_->offset += static_cast<int>(shape[shape.size() - 2]);
      }
    }
    ++chunk_index_;

    return CopyOrtOutput(ctc_output, vocab_size_, metadata_.onnx.output_type,
                         log_probs);
  } catch (const std::exception& e) {
    return Status::Internal(std::string("ONNX Runtime forward failed: ") +
                            e.what());
  }
#else
  (void)features;
  (void)log_probs;
  return Status::Unavailable("ONNX Runtime support is not compiled in");
#endif
}

}  // namespace wenet_sdk::internal
