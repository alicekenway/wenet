#ifndef ASR_SDK_STATUS_H_
#define ASR_SDK_STATUS_H_

#include <optional>
#include <sstream>
#include <string>
#include <utility>

#include "asr_sdk/export.h"

namespace asr_sdk {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument = 1,
  kNotFound = 2,
  kInternal = 3,
  kUnavailable = 4,
  kFailedPrecondition = 5,
};

class ASR_SDK_API Status {
 public:
  Status() = default;
  Status(StatusCode code, std::string message);

  static Status Ok();
  static Status InvalidArgument(std::string message);
  static Status NotFound(std::string message);
  static Status Internal(std::string message);
  static Status Unavailable(std::string message);
  static Status FailedPrecondition(std::string message);

  bool ok() const { return code_ == StatusCode::kOk; }
  StatusCode code() const { return code_; }
  const std::string& message() const { return message_; }
  std::string ToString() const;

 private:
  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

template <typename T>
class StatusOr {
 public:
  StatusOr(T value) : status_(Status::Ok()), value_(std::move(value)) {}
  StatusOr(Status status) : status_(std::move(status)) {}

  bool ok() const { return status_.ok(); }
  explicit operator bool() const { return ok(); }
  const Status& status() const { return status_; }

  T& value() & { return *value_; }
  const T& value() const& { return *value_; }
  T&& value() && { return std::move(*value_); }

 private:
  Status status_;
  std::optional<T> value_;
};

}  // namespace asr_sdk

#endif  // ASR_SDK_STATUS_H_
