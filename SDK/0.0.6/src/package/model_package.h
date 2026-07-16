#ifndef ASR_SDK_SRC_PACKAGE_MODEL_PACKAGE_H_
#define ASR_SDK_SRC_PACKAGE_MODEL_PACKAGE_H_

#include <filesystem>
#include <string>

#include "asr_sdk/config.h"
#include "asr_sdk/status.h"
#include "flashlight_decoder/flashlight_decoder_options.h"

namespace asr_sdk::internal {

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
  std::filesystem::path kenlm_bin;
  // Optional class-based contact LM.  The main KenLM remains kenlm_bin;
  // runtime contacts are available only for the explicit pattern-bias v1
  // format, a declared contact_class_word, and its metadata sidecar.
  std::filesystem::path contact_kenlm_bin;
  std::filesystem::path contact_lm_metadata;
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
  std::string contact_class_word = "<CONTACT>";
  bool contact_class_word_declared = false;
  bool contact_lm_declared = false;
  bool contact_lm_weight_declared = false;
  bool contact_lm_mode_declared = false;
  bool contact_lm_metadata_declared = false;
  bool contact_lm_accumulation_factor_declared = false;
  std::string contact_lm_mode;
  std::string contact_lm_metadata_format;
  double contact_lm_weight = 1.5;
  double contact_lm_accumulation_factor = 0.5;
  double contact_lm_max_bonus = 0.0;
  bool supports_runtime_contacts = false;
  flashlight_decoder::FlashlightDecoderOptions flashlight_options;
};

StatusOr<ModelPackage> LoadModelPackage(const EngineConfig& config);

}  // namespace asr_sdk::internal

#endif  // ASR_SDK_SRC_PACKAGE_MODEL_PACKAGE_H_
