#include "utils/json.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace asr_sdk::internal {
namespace {

size_t FindKey(const std::string& json, const std::string& key,
               size_t start = 0) {
  return json.find("\"" + key + "\"", start);
}

size_t FindValueStart(const std::string& json, size_t key_pos) {
  size_t colon = json.find(':', key_pos);
  if (colon == std::string::npos) {
    return std::string::npos;
  }
  ++colon;
  while (colon < json.size() &&
         std::isspace(static_cast<unsigned char>(json[colon]))) {
    ++colon;
  }
  return colon;
}

bool ParseJsonString(const std::string& json, size_t quote_pos,
                     std::string* value, size_t* next_pos) {
  if (quote_pos == std::string::npos || quote_pos >= json.size() ||
      json[quote_pos] != '"') {
    return false;
  }
  std::string out;
  bool escaped = false;
  for (size_t i = quote_pos + 1; i < json.size(); ++i) {
    const char c = json[i];
    if (escaped) {
      switch (c) {
        case '"':
        case '\\':
        case '/':
          out.push_back(c);
          break;
        case 'b':
          out.push_back('\b');
          break;
        case 'f':
          out.push_back('\f');
          break;
        case 'n':
          out.push_back('\n');
          break;
        case 'r':
          out.push_back('\r');
          break;
        case 't':
          out.push_back('\t');
          break;
        default:
          out.push_back(c);
          break;
      }
      escaped = false;
    } else if (c == '\\') {
      escaped = true;
    } else if (c == '"') {
      if (value != nullptr) {
        *value = std::move(out);
      }
      if (next_pos != nullptr) {
        *next_pos = i + 1;
      }
      return true;
    } else {
      out.push_back(c);
    }
  }
  return false;
}

}  // namespace

JsonValue::JsonValue(std::nullptr_t) : value_(nullptr) {}
JsonValue::JsonValue(bool value) : value_(value) {}
JsonValue::JsonValue(double value) : value_(value) {}
JsonValue::JsonValue(std::string value) : value_(std::move(value)) {}
JsonValue::JsonValue(Array value) : value_(std::move(value)) {}
JsonValue::JsonValue(Object value) : value_(std::move(value)) {}
bool JsonValue::IsNull() const { return std::holds_alternative<std::nullptr_t>(value_); }
bool JsonValue::IsBool() const { return std::holds_alternative<bool>(value_); }
bool JsonValue::IsNumber() const { return std::holds_alternative<double>(value_); }
bool JsonValue::IsString() const { return std::holds_alternative<std::string>(value_); }
bool JsonValue::IsArray() const { return std::holds_alternative<Array>(value_); }
bool JsonValue::IsObject() const { return std::holds_alternative<Object>(value_); }
bool JsonValue::AsBool() const { return std::get<bool>(value_); }
double JsonValue::AsNumber() const { return std::get<double>(value_); }
const std::string& JsonValue::AsString() const { return std::get<std::string>(value_); }
const JsonValue::Array& JsonValue::AsArray() const { return std::get<Array>(value_); }
const JsonValue::Object& JsonValue::AsObject() const { return std::get<Object>(value_); }

namespace {

class StrictJsonParser {
 public:
  explicit StrictJsonParser(const std::string& input) : input_(input) {}

  StatusOr<JsonValue> Parse() {
    SkipSpace();
    auto value = ParseValue();
    if (!value.ok()) return value.status();
    SkipSpace();
    if (pos_ != input_.size()) return Error("trailing content");
    return std::move(value).value();
  }

 private:
  Status Error(const std::string& message) const {
    return Status::InvalidArgument("invalid JSON at byte " +
                                   std::to_string(pos_) + ": " + message);
  }

  void SkipSpace() {
    while (pos_ < input_.size() &&
           std::isspace(static_cast<unsigned char>(input_[pos_]))) ++pos_;
  }

  bool Consume(char c) {
    SkipSpace();
    if (pos_ < input_.size() && input_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }

  StatusOr<JsonValue> ParseValue() {
    SkipSpace();
    if (pos_ >= input_.size()) return Error("expected value");
    const char c = input_[pos_];
    if (c == '{') return ParseObject();
    if (c == '[') return ParseArray();
    if (c == '"') {
      auto string = ParseString();
      if (!string.ok()) return string.status();
      return JsonValue(std::move(string).value());
    }
    if (c == 't' && input_.compare(pos_, 4, "true") == 0) {
      pos_ += 4;
      return JsonValue(true);
    }
    if (c == 'f' && input_.compare(pos_, 5, "false") == 0) {
      pos_ += 5;
      return JsonValue(false);
    }
    if (c == 'n' && input_.compare(pos_, 4, "null") == 0) {
      pos_ += 4;
      return JsonValue(nullptr);
    }
    return ParseNumber();
  }

  StatusOr<std::string> ParseString() {
    SkipSpace();
    if (pos_ >= input_.size() || input_[pos_] != '"') {
      return Error("expected string");
    }
    const size_t quote = pos_++;
    std::string value;
    size_t next = pos_;
    if (!ParseJsonString(input_, quote, &value, &next)) {
      return Error("unterminated string");
    }
    pos_ = next;
    return value;
  }

  StatusOr<JsonValue> ParseNumber() {
    const char* begin = input_.c_str() + pos_;
    char* end = nullptr;
    errno = 0;
    const double value = std::strtod(begin, &end);
    if (end == begin || errno == ERANGE || !std::isfinite(value)) {
      return Error("expected finite number");
    }
    pos_ += static_cast<size_t>(end - begin);
    return JsonValue(value);
  }

  StatusOr<JsonValue> ParseArray() {
    Consume('[');
    JsonValue::Array out;
    SkipSpace();
    if (Consume(']')) return JsonValue(std::move(out));
    while (true) {
      auto item = ParseValue();
      if (!item.ok()) return item.status();
      out.push_back(std::move(item).value());
      if (Consume(']')) break;
      if (!Consume(',')) return Error("expected ',' or ']'");
    }
    return JsonValue(std::move(out));
  }

  StatusOr<JsonValue> ParseObject() {
    Consume('{');
    JsonValue::Object out;
    SkipSpace();
    if (Consume('}')) return JsonValue(std::move(out));
    while (true) {
      auto key = ParseString();
      if (!key.ok()) return key.status();
      if (!Consume(':')) return Error("expected ':'");
      auto value = ParseValue();
      if (!value.ok()) return value.status();
      auto inserted = out.emplace(std::move(key).value(), std::move(value).value());
      if (!inserted.second) return Error("duplicate object key: " + inserted.first->first);
      if (Consume('}')) break;
      if (!Consume(',')) return Error("expected ',' or '}'");
    }
    return JsonValue(std::move(out));
  }

  const std::string& input_;
  size_t pos_ = 0;
};

}  // namespace

StatusOr<JsonValue> ParseJson(const std::string& json) {
  return StrictJsonParser(json).Parse();
}

std::string JsonEscape(const std::string& input) {
  std::string out;
  out.reserve(input.size() + 8);
  for (char c : input) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\b':
        out += "\\b";
        break;
      case '\f':
        out += "\\f";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\r':
        out += "\\r";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          out += ' ';
        } else {
          out += c;
        }
        break;
    }
  }
  return out;
}

std::string JsonUnescape(const std::string& input) {
  std::string wrapped = "\"" + input + "\"";
  std::string value;
  ParseJsonString(wrapped, 0, &value, nullptr);
  return value;
}

std::vector<std::string> ExtractJsonStringValues(const std::string& json,
                                                  const std::string& key) {
  std::vector<std::string> values;
  size_t pos = 0;
  while ((pos = FindKey(json, key, pos)) != std::string::npos) {
    size_t value_start = FindValueStart(json, pos);
    std::string value;
    size_t next = pos + key.size() + 2;
    if (ParseJsonString(json, value_start, &value, &next)) {
      values.push_back(std::move(value));
    }
    pos = next;
  }
  return values;
}

std::string FindJsonStringValue(const std::string& json,
                                const std::string& key,
                                const std::string& default_value) {
  auto values = ExtractJsonStringValues(json, key);
  if (values.empty()) {
    return default_value;
  }
  return values.front();
}

int FindJsonIntValue(const std::string& json, const std::string& key,
                     int default_value) {
  const size_t key_pos = FindKey(json, key);
  const size_t start = FindValueStart(json, key_pos);
  if (start == std::string::npos || start >= json.size()) {
    return default_value;
  }
  char* end = nullptr;
  const long value = std::strtol(json.c_str() + start, &end, 10);
  if (end == json.c_str() + start) {
    return default_value;
  }
  return static_cast<int>(value);
}

double FindJsonDoubleValue(const std::string& json, const std::string& key,
                           double default_value) {
  const size_t key_pos = FindKey(json, key);
  const size_t start = FindValueStart(json, key_pos);
  if (start == std::string::npos || start >= json.size()) {
    return default_value;
  }
  char* end = nullptr;
  const double value = std::strtod(json.c_str() + start, &end);
  if (end == json.c_str() + start) {
    return default_value;
  }
  return value;
}

bool FindJsonBoolValue(const std::string& json, const std::string& key,
                       bool default_value) {
  const size_t key_pos = FindKey(json, key);
  const size_t start = FindValueStart(json, key_pos);
  if (start == std::string::npos) {
    return default_value;
  }
  if (json.compare(start, 4, "true") == 0) {
    return true;
  }
  if (json.compare(start, 5, "false") == 0) {
    return false;
  }
  return default_value;
}

}  // namespace asr_sdk::internal
