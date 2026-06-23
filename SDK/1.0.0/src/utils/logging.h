#ifndef WENET_SDK_SRC_UTILS_LOGGING_H_
#define WENET_SDK_SRC_UTILS_LOGGING_H_

#include <iosfwd>
#include <sstream>

namespace wenet_sdk::internal {

enum class LogLevel {
  kError = 0,
  kWarn = 1,
  kInfo = 2,
  kDebug = 3,
  kTrace = 4,
};

void SetLogLevel(LogLevel level);
LogLevel GetLogLevel();

class LogMessage {
 public:
  LogMessage(LogLevel level, const char* file, int line);
  ~LogMessage();

  std::ostream& stream() { return stream_; }

 private:
  LogLevel level_;
  const char* file_;
  int line_;
  std::ostringstream stream_;
};

}  // namespace wenet_sdk::internal

#define WENETSDK_LOG(level)                                                    \
  ::wenet_sdk::internal::LogMessage(                                           \
      ::wenet_sdk::internal::LogLevel::k##level, __FILE__, __LINE__)           \
      .stream()

#endif  // WENET_SDK_SRC_UTILS_LOGGING_H_
