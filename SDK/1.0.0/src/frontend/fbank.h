#ifndef WENET_SDK_SRC_FRONTEND_FBANK_H_
#define WENET_SDK_SRC_FRONTEND_FBANK_H_

#include <vector>

#include "frontend/window.h"

namespace wenet_sdk::internal {

struct FbankOptions {
  int sample_rate = 16000;
  int num_bins = 80;
  int frame_length_ms = 25;
  int frame_shift_ms = 10;
  float low_freq = 20.0f;
  float high_freq = 0.0f;
  float preemphasis_coeff = 0.97f;
  float log_floor = 1.0e-10f;
  float waveform_scale = 32768.0f;
  bool remove_dc_offset = true;
  WindowType window_type = WindowType::kPovey;
};

class Fbank {
 public:
  explicit Fbank(FbankOptions options);

  std::vector<float> Compute(const std::vector<float>& frame) const;

  int sample_rate() const { return options_.sample_rate; }
  int feature_dim() const { return options_.num_bins; }
  int frame_length_samples() const { return frame_length_samples_; }
  int frame_shift_samples() const { return frame_shift_samples_; }

 private:
  void BuildMelBanks();

  FbankOptions options_;
  int frame_length_samples_ = 0;
  int frame_shift_samples_ = 0;
  int num_fft_bins_ = 0;
  std::vector<float> window_;
  std::vector<std::vector<float>> mel_banks_;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_FRONTEND_FBANK_H_
