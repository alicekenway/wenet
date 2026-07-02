#ifndef ASR_SDK_SRC_SHERPA_ONNX_WENET_WHISPER_FEATURE_EXTRACTOR_H_
#define ASR_SDK_SRC_SHERPA_ONNX_WENET_WHISPER_FEATURE_EXTRACTOR_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace asr_sdk::internal::sherpa_onnx_wenet {

enum class ZipformerFeatureType {
  kKaldi = 0,
  kNemo,
  kWhisper,
};

ZipformerFeatureType ParseZipformerFeatureType(const std::string& value);
const char* ZipformerFeatureTypeName(ZipformerFeatureType type);

struct WhisperFeatureOptions {
  ZipformerFeatureType feature_type = ZipformerFeatureType::kWhisper;
  int sample_rate = 16000;
  int feature_dim = 80;
  int frame_length_samples = 400;
  int frame_shift_samples = 160;
  int initial_padding_ms = 300;
  int final_padding_ms = 800;
};

class WhisperFeatureExtractor {
 public:
  explicit WhisperFeatureExtractor(WhisperFeatureOptions options);

  void ExtractPcm16(const int16_t* samples, size_t num_samples,
                    std::vector<std::vector<float>>* features) const;

 private:
  WhisperFeatureOptions options_;
};

}  // namespace asr_sdk::internal::sherpa_onnx_wenet

#endif  // ASR_SDK_SRC_SHERPA_ONNX_WENET_WHISPER_FEATURE_EXTRACTOR_H_
