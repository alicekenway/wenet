#ifndef ASR_SDK_SRC_UTILS_SLOT_FILE_H_
#define ASR_SDK_SRC_UTILS_SLOT_FILE_H_

#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>
#include <string>

#include "asr_sdk/decode_context.h"
#include "asr_sdk/status.h"

namespace asr_sdk::internal {

inline std::string TrimSlotLine(std::string value) {
  auto ok = [](unsigned char c) { return !std::isspace(c); };
  value.erase(value.begin(), std::find_if(value.begin(), value.end(), ok));
  value.erase(std::find_if(value.rbegin(), value.rend(), ok).base(), value.end());
  return value;
}

inline Status LoadSlotSectionFile(const std::string& path,
                                  DecodeContextConfig* config) {
  if (config == nullptr) return Status::InvalidArgument("slot config is null");
  std::ifstream input(path);
  if (!input) return Status::NotFound("failed to open slot file: " + path);
  SlotClass* current = nullptr;
  std::set<std::string> seen;
  std::string line;
  int line_number = 0;
  while (std::getline(input, line)) {
    ++line_number;
    line = TrimSlotLine(std::move(line));
    if (line.empty()) continue;
    if (line.front() == '#') {
      const std::string token = line.substr(1);
      if (token.size() < 3 || token.front() != '<' || token.back() != '>' ||
          token.find_first_of(" \t\r\n") != std::string::npos) {
        return Status::InvalidArgument("invalid slot header at line " +
                                       std::to_string(line_number));
      }
      auto it = std::find_if(config->slots.begin(), config->slots.end(),
                             [&](const SlotClass& item) {
                               return item.slot_token == token;
                             });
      if (it == config->slots.end()) {
        config->slots.push_back(SlotClass{});
        it = std::prev(config->slots.end());
        it->slot_token = token;
      }
      current = &*it;
      continue;
    }
    if (current == nullptr) {
      return Status::InvalidArgument("slot value before first header at line " +
                                     std::to_string(line_number));
    }
    if (!seen.insert(current->slot_token + "\x1f" + line).second) continue;
    SlotValueEntry value;
    value.value_id = line;
    value.display_name = line;
    SlotSpokenForm form;
    form.text = line;
    value.spoken_forms.push_back(std::move(form));
    current->values.push_back(std::move(value));
  }
  return Status::Ok();
}

}  // namespace asr_sdk::internal
#endif
