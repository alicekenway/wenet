#include "sdk/asr_stream_internal.h"

namespace asr_sdk::internal {
namespace {

class AsrStreamImpl final : public AsrStream {
 public:
  explicit AsrStreamImpl(std::unique_ptr<WenetStreamAdapter> adapter)
      : adapter_(std::move(adapter)) {}

  Status AcceptPcm16(const int16_t* samples, size_t num_samples,
                     int sample_rate) override {
    return adapter_->AcceptPcm16(samples, num_samples, sample_rate);
  }

  bool DecodeReady() const override { return adapter_->DecodeReady(); }
  Status Decode() override { return adapter_->Decode(); }
  AsrResult GetResult() const override { return adapter_->GetResult(); }
  AsrResult GetFinalResult() override { return adapter_->GetFinalResult(); }
  Status SetInputFinished() override { return adapter_->SetInputFinished(); }
  Status Reset() override { return adapter_->Reset(); }

 private:
  std::unique_ptr<WenetStreamAdapter> adapter_;
};

}  // namespace

std::unique_ptr<AsrStream> MakeAsrStream(
    std::unique_ptr<WenetStreamAdapter> adapter) {
  return std::unique_ptr<AsrStream>(new AsrStreamImpl(std::move(adapter)));
}

}  // namespace asr_sdk::internal
