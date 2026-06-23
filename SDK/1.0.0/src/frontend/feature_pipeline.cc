#include "frontend/feature_pipeline.h"

#include <algorithm>
#include <cstdint>

#include "frontend/resampler.h"

namespace wenet_sdk::internal {

namespace {

FbankOptions ToFbankOptions(const FeaturePipelineConfig& config) {
  FbankOptions options;
  options.sample_rate = config.sample_rate;
  options.num_bins = config.feature_dim;
  options.frame_length_ms = config.frame_length_ms;
  options.frame_shift_ms = config.frame_shift_ms;
  options.waveform_scale = config.waveform_scale;
  return options;
}

}  // namespace

FeaturePipeline::FeaturePipeline(FeaturePipelineConfig config)
    : config_(config), fbank_(ToFbankOptions(config)) {}

Status FeaturePipeline::LoadCmvnIfPresent(const std::filesystem::path& path) {
  return cmvn_.LoadIfPresent(path, config_.feature_dim);
}

Status FeaturePipeline::AcceptWaveform(int sample_rate, const float* samples,
                                       size_t n) {
  if (samples == nullptr && n > 0) {
    return Status::InvalidArgument("samples is null");
  }
  if (n == 0) {
    return Status::OK();
  }
  if (sample_rate <= 0) {
    return Status::InvalidArgument("sample rate must be positive");
  }

  std::vector<float> resampled;
  const float* accepted_samples = samples;
  size_t accepted_size = n;
  if (sample_rate != config_.sample_rate) {
    resampled = LinearResample(std::vector<float>(samples, samples + n),
                               sample_rate, config_.sample_rate);
    accepted_samples = resampled.data();
    accepted_size = resampled.size();
  }

  std::lock_guard<std::mutex> lock(mutex_);
  if (input_finished_) {
    return Status::InvalidArgument("cannot accept waveform after finish");
  }
  samples_.insert(samples_.end(), accepted_samples,
                  accepted_samples + accepted_size);
  ExtractAvailableFramesLocked();
  return Status::OK();
}

Status FeaturePipeline::AcceptWaveform(int sample_rate, const int16_t* samples,
                                       size_t n) {
  if (samples == nullptr && n > 0) {
    return Status::InvalidArgument("samples is null");
  }
  std::vector<float> converted(n);
  for (size_t i = 0; i < n; ++i) {
    converted[i] = static_cast<float>(samples[i]) / 32768.0f;
  }
  return AcceptWaveform(sample_rate, converted.data(), converted.size());
}

bool FeaturePipeline::DecodeReady() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return static_cast<int>(frames_.size()) >= config_.chunk_size ||
         (input_finished_ && !frames_.empty());
}

bool FeaturePipeline::ReadChunk(std::vector<std::vector<float>>* features) {
  if (features == nullptr) {
    return false;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  const int target =
      input_finished_ ? static_cast<int>(frames_.size()) : config_.chunk_size;
  const int n = std::min<int>(target, static_cast<int>(frames_.size()));
  if (n <= 0) {
    features->clear();
    return false;
  }
  features->clear();
  features->reserve(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    features->push_back(std::move(frames_.front()));
    frames_.pop_front();
  }
  return true;
}

void FeaturePipeline::SetInputFinished() {
  std::lock_guard<std::mutex> lock(mutex_);
  input_finished_ = true;
  if (!samples_.empty()) {
    samples_.resize(static_cast<size_t>(fbank_.frame_length_samples()), 0.0f);
    ExtractAvailableFramesLocked();
    samples_.clear();
  }
}

void FeaturePipeline::Reset() {
  std::lock_guard<std::mutex> lock(mutex_);
  samples_.clear();
  frames_.clear();
  input_finished_ = false;
}

int FeaturePipeline::queued_frames() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return static_cast<int>(frames_.size());
}

bool FeaturePipeline::input_finished() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return input_finished_;
}

void FeaturePipeline::ExtractAvailableFramesLocked() {
  const int frame_length = fbank_.frame_length_samples();
  const int frame_shift = fbank_.frame_shift_samples();
  while (static_cast<int>(samples_.size()) >= frame_length) {
    std::vector<float> frame(samples_.begin(),
                             samples_.begin() + frame_length);
    auto feat = fbank_.Compute(frame);
    cmvn_.Apply(&feat);
    frames_.push_back(std::move(feat));
    samples_.erase(samples_.begin(), samples_.begin() + frame_shift);
  }
}

}  // namespace wenet_sdk::internal
