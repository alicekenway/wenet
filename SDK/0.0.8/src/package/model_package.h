#ifndef ASR_SDK_SRC_PACKAGE_MODEL_PACKAGE_H_
#define ASR_SDK_SRC_PACKAGE_MODEL_PACKAGE_H_

#include <filesystem>
#include <string>
#include <vector>

#include "asr_sdk/config.h"
#include "asr_sdk/status.h"
#include "flashlight_decoder/flashlight_decoder_options.h"

namespace asr_sdk::internal {

enum class FixedLmType { kNgram, kBias };

struct FixedLmConfig {
  std::string filename;
  std::filesystem::path path;
  FixedLmType type = FixedLmType::kNgram;
  double weight = 1.0;
  bool clip = true;
  double clip_lower = 0.0;
  double clip_upper = 0.0;
  double accumulation_factor = 0.5;
  std::vector<std::string> slots;
};

struct ModelPackage {
  std::filesystem::path root;
  std::filesystem::path manifest;
  std::filesystem::path runtime_dir;
  std::filesystem::path encoder_onnx;
  std::filesystem::path ctc_onnx;
  std::filesystem::path decoder_onnx;
  std::filesystem::path units_txt;
  std::filesystem::path tokens_txt;
  std::filesystem::path words_txt;
  std::filesystem::path sherpa_ctc_onnx;
  std::filesystem::path lexicon_txt;
  std::filesystem::path lm_search_json;
  std::vector<FixedLmConfig> fixed_lms;
  std::filesystem::path output_mapping_txt;
  std::filesystem::path final_output_mapping_txt;
  std::filesystem::path tlg_fst;
  bool has_manifest = false;
  bool has_wfst = false;
  bool has_flashlight_decoder = false;
  bool debug = false;
  int sample_rate = 16000;
  int chunk_size = 16;
  int num_left_chunks = 16;
  int nbest = 1;
  bool enable_continuous_decoding = true;
  bool enable_timestamps = false;
  std::string language = "chs";
  std::string decoder_type = "wenet_wfst";
  std::string feature_type = "whisper";
  std::string blank_token = "<blk>";
  std::string sil_token = "▁";
  std::string unk_word = "<unk>";
  double length_penalty = 0.0;
  bool supports_runtime_slots = false;
  flashlight_decoder::FlashlightDecoderOptions flashlight_options;
};

StatusOr<ModelPackage> LoadModelPackage(const EngineConfig& config);

}  // namespace asr_sdk::internal

#endif  // ASR_SDK_SRC_PACKAGE_MODEL_PACKAGE_H_
