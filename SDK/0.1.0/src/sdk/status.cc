#include "asr_sdk/status.h"

namespace asr_sdk {

Status::Status(StatusCode code, std::string message)
    : code_(code), message_(std::move(message)) {}

Status Status::Ok() { return Status(); }

Status Status::InvalidArgument(std::string message) {
  return Status(StatusCode::kInvalidArgument, std::move(message));
}

Status Status::NotFound(std::string message) {
  return Status(StatusCode::kNotFound, std::move(message));
}

Status Status::Internal(std::string message) {
  return Status(StatusCode::kInternal, std::move(message));
}

Status Status::Unavailable(std::string message) {
  return Status(StatusCode::kUnavailable, std::move(message));
}

Status Status::FailedPrecondition(std::string message) {
  return Status(StatusCode::kFailedPrecondition, std::move(message));
}

std::string Status::ToString() const {
  if (ok()) {
    return "OK";
  }
  std::ostringstream out;
  out << static_cast<int>(code_) << ": " << message_;
  return out.str();
}

}  // namespace asr_sdk
