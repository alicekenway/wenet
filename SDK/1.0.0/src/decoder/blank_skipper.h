#ifndef WENET_SDK_SRC_DECODER_BLANK_SKIPPER_H_
#define WENET_SDK_SRC_DECODER_BLANK_SKIPPER_H_

#include <vector>

namespace wenet_sdk::internal {

struct BlankSkipperOptions {
  int blank_id = 0;
  float blank_skip_thresh = 1.01f;
  bool enabled = false;
};

class BlankSkipper {
 public:
  explicit BlankSkipper(BlankSkipperOptions options);

  void Reset();
  std::vector<std::vector<float>> Filter(
      const std::vector<std::vector<float>>& log_probs);

  int frames_seen() const { return frames_seen_; }
  int frames_skipped() const { return frames_skipped_; }

 private:
  BlankSkipperOptions options_;
  int frames_seen_ = 0;
  int frames_skipped_ = 0;
  bool skipped_previous_ = false;
  std::vector<float> previous_skipped_;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_DECODER_BLANK_SKIPPER_H_
