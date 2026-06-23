#ifndef WENET_SDK_SRC_POSTPROCESS_ENDPOINT_H_
#define WENET_SDK_SRC_POSTPROCESS_ENDPOINT_H_

namespace wenet_sdk::internal {

struct EndpointOptions {
  int endpoint_silence_ms = 800;
  int max_segment_ms = 20000;
  int frame_shift_ms = 10;
};

class Endpoint {
 public:
  explicit Endpoint(EndpointOptions options);

  void Reset();
  void AdvanceFrames(int frames, bool blank_heavy);
  bool IsEndpoint(bool input_finished) const;

 private:
  EndpointOptions options_;
  int total_frames_ = 0;
  int trailing_blank_frames_ = 0;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_POSTPROCESS_ENDPOINT_H_
