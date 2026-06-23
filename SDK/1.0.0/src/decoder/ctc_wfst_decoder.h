#ifndef WENET_SDK_SRC_DECODER_CTC_WFST_DECODER_H_
#define WENET_SDK_SRC_DECODER_CTC_WFST_DECODER_H_

#include <filesystem>
#include <memory>

#include "decoder/ctc_prefix_decoder.h"
#include "utils/status.h"

namespace wenet_sdk::internal {

struct CtcWfstDecoderOptions {
  int blank_id = 0;
  float beam = 16.0f;
  float lattice_beam = 10.0f;
  int max_active = 7000;
  int min_active = 200;
  float acoustic_scale = 1.0f;
  float lm_scale = 1.0f;
  float length_penalty = 0.0f;
  int nbest = 1;
};

class CtcWfstDecoder final : public StreamingDecoder {
 public:
  static Status ValidateGraphPath(const std::filesystem::path& path);

  CtcWfstDecoder(CtcWfstDecoderOptions options,
                 std::filesystem::path graph_path);
  ~CtcWfstDecoder() override;

  void Reset() override;
  void Advance(const std::vector<std::vector<float>>& log_probs) override;
  DecodeResult PartialResult() const override;
  DecodeResult Finalize() override;

 private:
  struct Impl;

  CtcPrefixDecoder fallback_;
  std::filesystem::path graph_path_;
  CtcWfstDecoderOptions options_;
  std::unique_ptr<Impl> impl_;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_DECODER_CTC_WFST_DECODER_H_
