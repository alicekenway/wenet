#include "utils/json.h"

#include <cctype>
#include <cstdlib>

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
