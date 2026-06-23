#ifndef WENET_SDK_SRC_FRONTEND_FEATURE_PIPELINE_H_
#define WENET_SDK_SRC_FRONTEND_FEATURE_PIPELINE_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <mutex>
#include <vector>

#include "frontend/cmvn.h"
#include "frontend/fbank.h"
#include "utils/status.h"

namespace wenet_sdk::internal {

struct FeaturePipelineConfig {
  int sample_rate = 16000;
  int feature_dim = 80;
  int frame_length_ms = 25;
  int frame_shift_ms = 10;
  int chunk_size = 16;
  float waveform_scale = 32768.0f;
};

class FeaturePipeline {
 public:
  explicit FeaturePipeline(FeaturePipelineConfig config);

  Status LoadCmvnIfPresent(const std::filesystem::path& path);
  Status AcceptWaveform(int sample_rate, const float* samples, size_t n);
  Status AcceptWaveform(int sample_rate, const int16_t* samples, size_t n);

  bool DecodeReady() const;
  bool ReadChunk(std::vector<std::vector<float>>* features);
  void SetInputFinished();
  void Reset();

  int feature_dim() const { return config_.feature_dim; }
  int sample_rate() const { return config_.sample_rate; }
  int queued_frames() const;
  bool input_finished() const;

 private:
  void ExtractAvailableFramesLocked();

  FeaturePipelineConfig config_;
  Fbank fbank_;
  Cmvn cmvn_;

  mutable std::mutex mutex_;
  std::vector<float> samples_;
  std::deque<std::vector<float>> frames_;
  bool input_finished_ = false;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_FRONTEND_FEATURE_PIPELINE_H_
