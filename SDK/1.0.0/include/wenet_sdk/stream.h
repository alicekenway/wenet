#ifndef WENET_SDK_STREAM_H_
#define WENET_SDK_STREAM_H_

#include <cstddef>

#include "wenet_sdk/result.h"

namespace wenet_sdk {

class Stream {
 public:
  virtual ~Stream() = default;

  virtual void AcceptWaveform(int sample_rate, const float* samples,
                              size_t n) = 0;

  virtual bool DecodeReady() const = 0;
  virtual void Decode() = 0;

  virtual AsrResult GetResult() const = 0;
  virtual AsrResult GetFinalResult() = 0;

  virtual void SetInputFinished() = 0;
  virtual void Reset() = 0;
};

}  // namespace wenet_sdk

#endif  // WENET_SDK_STREAM_H_
