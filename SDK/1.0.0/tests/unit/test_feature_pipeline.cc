#include <cassert>
#include <cmath>
#include <vector>

#include "frontend/feature_pipeline.h"

int main() {
  wenet_sdk::internal::FeaturePipelineConfig config;
  config.sample_rate = 16000;
  config.feature_dim = 16;
  config.chunk_size = 2;
  wenet_sdk::internal::FeaturePipeline pipeline(config);

  std::vector<float> samples(1000);
  for (size_t i = 0; i < samples.size(); ++i) {
    samples[i] = 0.1f * std::sin(2.0 * 3.14159265358979323846 * i / 32.0);
  }
  auto status = pipeline.AcceptWaveform(16000, samples.data(), samples.size());
  assert(status.ok());
  assert(pipeline.DecodeReady());
  std::vector<std::vector<float>> chunk;
  assert(pipeline.ReadChunk(&chunk));
  assert(chunk.size() == 2);
  assert(chunk[0].size() == 16);

  pipeline.Reset();
  std::vector<float> low_rate_samples(500);
  for (size_t i = 0; i < low_rate_samples.size(); ++i) {
    low_rate_samples[i] =
        0.1f * std::sin(2.0 * 3.14159265358979323846 * i / 16.0);
  }
  status = pipeline.AcceptWaveform(8000, low_rate_samples.data(),
                                   low_rate_samples.size());
  assert(status.ok());
  assert(pipeline.DecodeReady());
  assert(pipeline.ReadChunk(&chunk));
  assert(chunk.size() == 2);
  assert(chunk[0].size() == 16);

  status = pipeline.AcceptWaveform(0, low_rate_samples.data(),
                                   low_rate_samples.size());
  assert(!status.ok());
  return 0;
}
