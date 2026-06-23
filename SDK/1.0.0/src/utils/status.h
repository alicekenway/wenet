#ifndef WENET_SDK_SRC_UTILS_STATUS_H_
#define WENET_SDK_SRC_UTILS_STATUS_H_

#include <string>

namespace wenet_sdk::internal {

enum class StatusCode {
  kOk = 0,
  kInvalidArgument = 1,
  kNotFound = 2,
  kInternal = 3,
  kUnavailable = 4,
};

class Status {
 public:
  Status() = default;
  Status(StatusCode code, std::string message)
      : code_(code), message_(std::move(message)) {}

  static Status OK() { return Status(); }
  static Status InvalidArgument(std::string message) {
    return Status(StatusCode::kInvalidArgument, std::move(message));
  }
  static Status NotFound(std::string message) {
    return Status(StatusCode::kNotFound, std::move(message));
  }
  static Status Internal(std::string message) {
    return Status(StatusCode::kInternal, std::move(message));
  }
  static Status Unavailable(std::string message) {
    return Status(StatusCode::kUnavailable, std::move(message));
  }

  bool ok() const { return code_ == StatusCode::kOk; }
  StatusCode code() const { return code_; }
  const std::string& message() const { return message_; }

 private:
  StatusCode code_ = StatusCode::kOk;
  std::string message_;
};

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_UTILS_STATUS_H_
