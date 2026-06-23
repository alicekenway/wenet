#include "utils/logging.h"

#include <atomic>
#include <iostream>

namespace wenet_sdk::internal {
namespace {

std::atomic<int> g_log_level{static_cast<int>(LogLevel::kWarn)};

const char* LevelName(LogLevel level) {
  switch (level) {
    case LogLevel::kError:
      return "ERROR";
    case LogLevel::kWarn:
      return "WARN";
    case LogLevel::kInfo:
      return "INFO";
    case LogLevel::kDebug:
      return "DEBUG";
    case LogLevel::kTrace:
      return "TRACE";
  }
  return "UNKNOWN";
}

}  // namespace

void SetLogLevel(LogLevel level) {
  g_log_level.store(static_cast<int>(level));
}

LogLevel GetLogLevel() {
  return static_cast<LogLevel>(g_log_level.load());
}

LogMessage::LogMessage(LogLevel level, const char* file, int line)
    : level_(level), file_(file), line_(line) {}

LogMessage::~LogMessage() {
  if (static_cast<int>(level_) > g_log_level.load()) {
    return;
  }
  std::cerr << "[" << LevelName(level_) << "] " << file_ << ":" << line_
            << " " << stream_.str() << "\n";
}

}  // namespace wenet_sdk::internal
