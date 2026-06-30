#ifndef ASR_SDK_STREAM_H_
#define ASR_SDK_STREAM_H_

#include <cstddef>
#include <cstdint>

#include "asr_sdk/export.h"
#include "asr_sdk/result.h"
#include "asr_sdk/status.h"

namespace asr_sdk {

class ASR_SDK_API AsrStream {
 public:
  virtual ~AsrStream() = default;

  virtual Status AcceptPcm16(const int16_t* samples, size_t num_samples,
                             int sample_rate) = 0;
  virtual bool DecodeReady() const = 0;
  virtual Status Decode() = 0;
  virtual AsrResult GetResult() const = 0;
  virtual AsrResult GetFinalResult() = 0;
  virtual Status SetInputFinished() = 0;
  virtual Status Reset() = 0;
};

}  // namespace asr_sdk

#endif  // ASR_SDK_STREAM_H_
