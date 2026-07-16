#ifndef ASR_SDK_SRC_UTILS_JSON_H_
#define ASR_SDK_SRC_UTILS_JSON_H_

#include <string>
#include <vector>

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

}  // namespace asr_sdk::internal

#endif  // ASR_SDK_SRC_UTILS_JSON_H_
