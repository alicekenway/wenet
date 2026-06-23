#ifndef WENET_SDK_SRC_CORE_RECOGNIZER_H_
#define WENET_SDK_SRC_CORE_RECOGNIZER_H_

#include <memory>
#include <cstdint>

#include "core/engine_resources.h"
#include "core/result_builder.h"
#include "decoder/blank_skipper.h"
#include "decoder/decoder_interface.h"
#include "frontend/feature_pipeline.h"
#include "model/onnx_ctc_model.h"
#include "postprocess/endpoint.h"
#include "utils/status.h"

namespace wenet_sdk::internal {

class Recognizer {
 public:
  explicit Recognizer(std::shared_ptr<const EngineResources> resources);

  Status Init();
  Status AcceptWaveform(int sample_rate, const float* samples, size_t n);
  Status AcceptWaveform(int sample_rate, const int16_t* samples, size_t n);
  bool DecodeReady() const;
  Status Decode();
  void SetInputFinished();
  void Reset();

  AsrResult GetResult() const { return latest_result_; }
  AsrResult GetFinalResult();

 private:
  std::unique_ptr<StreamingDecoder> CreateDecoder() const;

  std::shared_ptr<const EngineResources> resources_;
  FeaturePipeline feature_pipeline_;
  OnnxCtcModel model_;
  BlankSkipper blank_skipper_;
  Endpoint endpoint_;
  std::unique_ptr<StreamingDecoder> decoder_;
  ResultBuilder result_builder_;
  AsrResult latest_result_;
  AsrResult final_result_;
  bool input_finished_ = false;
  bool finalized_ = false;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_CORE_RECOGNIZER_H_
