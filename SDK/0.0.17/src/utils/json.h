#ifndef ASR_SDK_SRC_UTILS_JSON_H_
#define ASR_SDK_SRC_UTILS_JSON_H_

#include <string>
#include <map>
#include <variant>
#include <vector>

#include "asr_sdk/status.h"

namespace asr_sdk::internal {

std::string JsonEscape(const std::string& input);
std::string JsonUnescape(const std::string& input);
std::vector<std::string> ExtractJsonStringValues(const std::string& json,
                                                  const std::string& key);
std::string FindJsonStringValue(const std::string& json,
                                const std::string& key,
                                const std::string& default_value);
int FindJsonIntValue(const std::string& json, const std::string& key,
                     int default_value);
double FindJsonDoubleValue(const std::string& json, const std::string& key,
                           double default_value);
bool FindJsonBoolValue(const std::string& json, const std::string& key,
                       bool default_value);

// Strict JSON value/parser used by package configuration.  The historical
// FindJson* helpers remain for flat legacy manifests, but must not be used for
// nested objects because they cannot validate structure or duplicate keys.
class JsonValue {
 public:
  using Object = std::map<std::string, JsonValue>;
  using Array = std::vector<JsonValue>;

  JsonValue() = default;
  explicit JsonValue(std::nullptr_t);
  explicit JsonValue(bool value);
  explicit JsonValue(double value);
  explicit JsonValue(std::string value);
  explicit JsonValue(Array value);
  explicit JsonValue(Object value);

  bool IsNull() const;
  bool IsBool() const;
  bool IsNumber() const;
  bool IsString() const;
  bool IsArray() const;
  bool IsObject() const;
  bool AsBool() const;
  double AsNumber() const;
  const std::string& AsString() const;
  const Array& AsArray() const;
  const Object& AsObject() const;

 private:
  std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value_;
};

StatusOr<JsonValue> ParseJson(const std::string& json);

}  // namespace asr_sdk::internal

#endif  // ASR_SDK_SRC_UTILS_JSON_H_
