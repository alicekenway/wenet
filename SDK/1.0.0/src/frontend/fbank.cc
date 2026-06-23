#include "frontend/fbank.h"

#include <algorithm>
#include <cmath>

namespace wenet_sdk::internal {
namespace {

double Pi() { return std::acos(-1.0); }

float HertzToMel(float hz) {
  return 1127.0f * std::log(1.0f + hz / 700.0f);
}

float MelToHertz(float mel) {
  return 700.0f * (std::exp(mel / 1127.0f) - 1.0f);
}

int NextPowerOfTwo(int value) {
  int n = 1;
  while (n < value) {
    n <<= 1;
  }
  return n;
}

}  // namespace

Fbank::Fbank(FbankOptions options) : options_(options) {
  frame_length_samples_ =
      options_.sample_rate * options_.frame_length_ms / 1000;
  frame_shift_samples_ = options_.sample_rate * options_.frame_shift_ms / 1000;
  const int fft_size = NextPowerOfTwo(frame_length_samples_);
  num_fft_bins_ = fft_size / 2 + 1;
  window_ = BuildWindow(frame_length_samples_, options_.window_type);
  BuildMelBanks();
}

void Fbank::BuildMelBanks() {
  mel_banks_.assign(static_cast<size_t>(options_.num_bins),
                    std::vector<float>(static_cast<size_t>(num_fft_bins_),
                                       0.0f));

  const float high_freq =
      options_.high_freq > 0.0f ? options_.high_freq
                                : static_cast<float>(options_.sample_rate) / 2;
  const float low_mel = HertzToMel(options_.low_freq);
  const float high_mel = HertzToMel(high_freq);
  std::vector<float> mel_points(static_cast<size_t>(options_.num_bins + 2));
  for (int i = 0; i < options_.num_bins + 2; ++i) {
    const float ratio = static_cast<float>(i) / (options_.num_bins + 1);
    mel_points[static_cast<size_t>(i)] = low_mel + ratio * (high_mel - low_mel);
  }

  std::vector<int> bins(static_cast<size_t>(options_.num_bins + 2));
  for (int i = 0; i < options_.num_bins + 2; ++i) {
    const float hz = MelToHertz(mel_points[static_cast<size_t>(i)]);
    bins[static_cast<size_t>(i)] =
        std::max(0, std::min(num_fft_bins_ - 1,
                             static_cast<int>(std::floor(
                                 (num_fft_bins_ - 1) * 2.0f * hz /
                                 options_.sample_rate))));
  }

  for (int mel = 0; mel < options_.num_bins; ++mel) {
    const int left = bins[static_cast<size_t>(mel)];
    const int center = bins[static_cast<size_t>(mel + 1)];
    const int right = bins[static_cast<size_t>(mel + 2)];
    for (int bin = left; bin < center; ++bin) {
      const float denom = static_cast<float>(std::max(1, center - left));
      mel_banks_[static_cast<size_t>(mel)][static_cast<size_t>(bin)] =
          (bin - left) / denom;
    }
    for (int bin = center; bin < right; ++bin) {
      const float denom = static_cast<float>(std::max(1, right - center));
      mel_banks_[static_cast<size_t>(mel)][static_cast<size_t>(bin)] =
          (right - bin) / denom;
    }
  }
}

std::vector<float> Fbank::Compute(const std::vector<float>& frame) const {
  std::vector<float> work = frame;
  work.resize(static_cast<size_t>(frame_length_samples_), 0.0f);
  for (float& sample : work) {
    sample *= options_.waveform_scale;
  }
  if (options_.remove_dc_offset && !work.empty()) {
    double mean = 0.0;
    for (float sample : work) {
      mean += sample;
    }
    mean /= static_cast<double>(work.size());
    for (float& sample : work) {
      sample -= static_cast<float>(mean);
    }
  }
  ApplyPreemphasis(&work, options_.preemphasis_coeff);
  ApplyWindow(&work, window_);

  const int fft_size = (num_fft_bins_ - 1) * 2;
  std::vector<float> power(static_cast<size_t>(num_fft_bins_), 0.0f);
  for (int k = 0; k < num_fft_bins_; ++k) {
    double real = 0.0;
    double imag = 0.0;
    for (int n = 0; n < frame_length_samples_; ++n) {
      const double angle = -2.0 * Pi() * k * n / fft_size;
      const double sample = work[static_cast<size_t>(n)];
      real += sample * std::cos(angle);
      imag += sample * std::sin(angle);
    }
    power[static_cast<size_t>(k)] =
        static_cast<float>(real * real + imag * imag);
  }

  std::vector<float> features(static_cast<size_t>(options_.num_bins), 0.0f);
  for (int mel = 0; mel < options_.num_bins; ++mel) {
    double energy = 0.0;
    for (int k = 0; k < num_fft_bins_; ++k) {
      energy += power[static_cast<size_t>(k)] *
                mel_banks_[static_cast<size_t>(mel)][static_cast<size_t>(k)];
    }
    features[static_cast<size_t>(mel)] =
        std::log(std::max(static_cast<float>(energy), options_.log_floor));
  }
  return features;
}

}  // namespace wenet_sdk::internal
