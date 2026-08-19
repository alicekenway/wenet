#ifndef ASR_SDK_SRC_UTILS_TIMER_H_
#define ASR_SDK_SRC_UTILS_TIMER_H_

#include <chrono>

namespace asr_sdk::internal {

class Timer {
 public:
  Timer();
  void Reset();
  double ElapsedSeconds() const;
  double ElapsedMilliseconds() const;

 private:
  std::chrono::steady_clock::time_point start_;
};

}  // namespace asr_sdk::internal

#endif  // ASR_SDK_SRC_UTILS_TIMER_H_
