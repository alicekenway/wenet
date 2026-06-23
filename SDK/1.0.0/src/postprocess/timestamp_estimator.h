#ifndef WENET_SDK_SRC_POSTPROCESS_TIMESTAMP_ESTIMATOR_H_
#define WENET_SDK_SRC_POSTPROCESS_TIMESTAMP_ESTIMATOR_H_

namespace wenet_sdk::internal {

class TimestampEstimator {
 public:
  TimestampEstimator(int frame_shift_ms, int subsampling_rate);

  float StartMs(int decoder_frame) const;
  float EndMs(int decoder_frame) const;

 private:
  int frame_shift_ms_ = 10;
  int subsampling_rate_ = 4;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_POSTPROCESS_TIMESTAMP_ESTIMATOR_H_
