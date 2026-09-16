#ifndef ASR_SDK_SRC_SHERPA_ONNX_WENET_CTC_ONNX_SESSION_H_
#define ASR_SDK_SRC_SHERPA_ONNX_WENET_CTC_ONNX_SESSION_H_

#include <algorithm>
#include <string>

#include "asr_sdk/config.h"
#include "onnxruntime_cxx_api.h"  // NOLINT

namespace asr_sdk::internal::sherpa_onnx_wenet {

// Owns the complete lifetime required by one ONNX Runtime session.  The
// backend factory reads model metadata through this object and then transfers
// the same object to the selected backend resource, avoiding a probe session.
class CtcOnnxSession {
 public:
  static OrtLoggingLevel ToOrtLevel(OrtLogLevel level) {
    switch (level) {
      case OrtLogLevel::kVerbose: return ORT_LOGGING_LEVEL_VERBOSE;
      case OrtLogLevel::kInfo: return ORT_LOGGING_LEVEL_INFO;
      case OrtLogLevel::kWarning: return ORT_LOGGING_LEVEL_WARNING;
      default: return ORT_LOGGING_LEVEL_ERROR;
    }
  }

  CtcOnnxSession(const std::string& model_path, int num_threads,
                 OrtLogLevel log_level = OrtLogLevel::kError,
                 bool prepacking = false, bool device_initializers = false)
      : env_(ToOrtLevel(log_level), "ctc_onnx_backend") {
    options_.SetIntraOpNumThreads(std::max(1, num_threads));
    options_.SetInterOpNumThreads(1);
    options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    options_.SetLogSeverityLevel(ToOrtLevel(log_level));
    options_.AddConfigEntry("session.disable_prepacking", prepacking ? "0" : "1");
    options_.AddConfigEntry("session.use_device_allocator_for_initializers", device_initializers ? "1" : "0");
    session_ = Ort::Session(env_, model_path.c_str(), options_);
  }

  Ort::Session& Session() { return session_; }
  const Ort::Session& Session() const { return session_; }

 private:
  Ort::Env env_;
  Ort::SessionOptions options_;
  Ort::Session session_{nullptr};
};

}  // namespace asr_sdk::internal::sherpa_onnx_wenet

#endif  // ASR_SDK_SRC_SHERPA_ONNX_WENET_CTC_ONNX_SESSION_H_
