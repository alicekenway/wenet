#include <cassert>
#include <cmath>
#include <vector>

#include "wenet_sdk/asr_engine.h"

int main() {
  wenet_sdk::EngineConfig config;
#ifdef WENETSDK_ENABLE_ONNX
  config.model_dir = "tests/fixtures/onnx_toy";
#else
  config.model_dir = "model_example";
#endif
  config.chunk_size = 4;
  auto engine = wenet_sdk::AsrEngine::Create(config);
  assert(engine);
  auto stream = engine->CreateStream();
  assert(stream);

  const int sample_rate = 16000;
  std::vector<float> pcm(3200);
  for (size_t i = 0; i < pcm.size(); ++i) {
    pcm[i] = 0.2f * std::sin(2.0 * 3.14159265358979323846 * 440.0 *
                             static_cast<double>(i) / sample_rate);
  }
  stream->AcceptWaveform(sample_rate, pcm.data(), pcm.size());
  assert(stream->DecodeReady());
  while (stream->DecodeReady()) {
    stream->Decode();
  }
  stream->SetInputFinished();
  while (stream->DecodeReady()) {
    stream->Decode();
  }
  const auto final_result = stream->GetFinalResult();
  assert(final_result.is_final);
  return 0;
}
