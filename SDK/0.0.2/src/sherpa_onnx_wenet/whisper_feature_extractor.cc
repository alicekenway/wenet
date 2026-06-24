#include "sherpa_onnx_wenet/whisper_feature_extractor.h"

#include <algorithm>
#include <stdexcept>

#include "frontend/fbank.h"

namespace asr_sdk::internal::sherpa_onnx_wenet {

WhisperFeatureExtractor::WhisperFeatureExtractor(WhisperFeatureOptions options)
    : options_(options) {
  if (options_.sample_rate != 16000) {
    throw std::runtime_error("Zipformer CTC currently supports 16 kHz only");
  }
  if (options_.feature_dim != 80) {
    throw std::runtime_error("Zipformer CTC currently supports 80-bin features");
  }
}

void WhisperFeatureExtractor::ExtractPcm16(
    const int16_t* samples, size_t num_samples,
    std::vector<std::vector<float>>* features) const {
  if (features == nullptr) {
    throw std::runtime_error("feature output is null");
  }
  if (samples == nullptr && num_samples > 0) {
    throw std::runtime_error("PCM input is null");
  }

  const int initial_padding_samples =
      options_.sample_rate * options_.initial_padding_ms / 1000;
  const int final_padding_samples =
      options_.sample_rate * options_.final_padding_ms / 1000;
  std::vector<float> pcm;
  pcm.reserve(static_cast<size_t>(initial_padding_samples) + num_samples +
              static_cast<size_t>(final_padding_samples));
  pcm.insert(pcm.end(), static_cast<size_t>(initial_padding_samples), 0.0f);
  for (size_t i = 0; i < num_samples; ++i) {
    pcm.push_back(static_cast<float>(samples[i]));
  }
  pcm.insert(pcm.end(), static_cast<size_t>(final_padding_samples), 0.0f);

  wenet::Fbank fbank(options_.feature_dim, options_.sample_rate,
                     options_.frame_length_samples,
                     options_.frame_shift_samples,
                     /*low_freq=*/0.0f,
                     /*pre_emphasis=*/false,
                     /*scale_input_to_unit=*/true,
                     /*log_floor=*/1.0e-10f, wenet::LogBase::kBase10,
                     wenet::WindowType::kHanning, wenet::MelType::kSlaney,
                     wenet::NormalizationType::kWhisper);
  fbank.set_dither(0.0f);
  fbank.Compute(pcm, features);
}

}  // namespace asr_sdk::internal::sherpa_onnx_wenet
