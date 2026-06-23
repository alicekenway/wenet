#ifndef WENET_SDK_SRC_MODEL_MODEL_METADATA_H_
#define WENET_SDK_SRC_MODEL_MODEL_METADATA_H_

#include <filesystem>
#include <string>

#include "utils/status.h"
#include "wenet_sdk/config.h"

namespace wenet_sdk::internal {

struct OnnxMetadata {
  std::string encoder = "encoder.onnx";
  std::string ctc = "ctc.onnx";
  std::string output_type = "log_prob";
  std::string chunk_input_name = "chunk";
  std::string offset_input_name = "offset";
  std::string att_cache_input_name = "att_cache";
  std::string cnn_cache_input_name = "cnn_cache";
  std::string log_probs_output_name = "log_probs";
};

struct VocabMetadata {
  std::string tokens = "tokens.txt";
  std::string words = "words.txt";
  int blank_id = 0;
  int sos_id = -1;
  int eos_id = -1;
};

struct DecoderMetadata {
  DecoderType type = DecoderType::kGreedyCtc;
  std::string graph = "TLG.fst";
  float beam = 16.0f;
  float lattice_beam = 10.0f;
  int max_active = 7000;
  int min_active = 200;
  float acoustic_scale = 1.0f;
  float lm_scale = 1.0f;
  float length_penalty = 0.0f;
  float blank_skip_thresh = 0.98f;
  int nbest = 1;
};

struct StreamingMetadata {
  int chunk_size = 16;
  int num_left_chunks = 4;
  int endpoint_silence_ms = 800;
  int max_segment_ms = 20000;
};

struct RuntimeMetadata {
  int intra_op_num_threads = 1;
  int inter_op_num_threads = 1;
  bool enable_profiling = false;
};

struct PostprocessMetadata {
  bool lowercase = true;
  bool remove_bpe_marker = true;
  std::string language_type = "indo_european";
};

struct ModelMetadata {
  std::filesystem::path model_dir;
  int sdk_model_version = 1;
  std::string model_type = "wenet_ctc_streaming_onnx";
  int sample_rate = 16000;
  int feature_dim = 80;
  int frame_length_ms = 25;
  int frame_shift_ms = 10;
  int subsampling_rate = 4;
  float waveform_scale = 32768.0f;

  OnnxMetadata onnx;
  VocabMetadata vocab;
  DecoderMetadata decoder;
  StreamingMetadata streaming;
  RuntimeMetadata runtime;
  PostprocessMetadata postprocess;

  std::filesystem::path Resolve(const std::string& relative_path) const {
    return model_dir / relative_path;
  }
};

Status LoadModelMetadata(const std::filesystem::path& model_dir,
                         ModelMetadata* metadata);
Status ValidateModelPackageFiles(const ModelMetadata& metadata,
                                 bool require_runtime_models);
std::string ModelSummary(const ModelMetadata& metadata);
DecoderType ParseDecoderType(const std::string& value);
std::string DecoderTypeName(DecoderType type);

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_MODEL_MODEL_METADATA_H_
