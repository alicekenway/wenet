#include "utils/timer.h"

namespace asr_sdk::internal {

Timer::Timer() { Reset(); }

void Timer::Reset() { start_ = std::chrono::steady_clock::now(); }

double Timer::ElapsedSeconds() const {
  const auto elapsed = std::chrono::steady_clock::now() - start_;
  return std::chrono::duration<double>(elapsed).count();
}

double Timer::ElapsedMilliseconds() const {
  return ElapsedSeconds() * 1000.0;
}

}  // namespace asr_sdk::internal
