#ifndef ASR_SDK_SRC_PACKAGE_MODEL_PACKAGE_H_
#define ASR_SDK_SRC_PACKAGE_MODEL_PACKAGE_H_

#include <filesystem>
#include <string>

#include "asr_sdk/config.h"
#include "asr_sdk/status.h"

namespace asr_sdk::internal {

struct ModelPackage {
  std::filesystem::path root;
  std::filesystem::path manifest;
  std::filesystem::path runtime_dir;
  std::filesystem::path encoder_onnx;
  std::filesystem::path ctc_onnx;
  std::filesystem::path decoder_onnx;
  std::filesystem::path units_txt;
  std::filesystem::path words_txt;
  std::filesystem::path tlg_fst;
  bool has_manifest = false;
  bool has_wfst = false;
  int sample_rate = 16000;
  int chunk_size = 16;
  int num_left_chunks = 16;
  int nbest = 1;
  bool enable_continuous_decoding = true;
  bool enable_timestamps = false;
  std::string language = "chs";
};

StatusOr<ModelPackage> LoadModelPackage(const EngineConfig& config);

}  // namespace asr_sdk::internal

#endif  // ASR_SDK_SRC_PACKAGE_MODEL_PACKAGE_H_
