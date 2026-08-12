#ifndef ASR_SDK_SRC_UTILS_FILE_UTILS_H_
#define ASR_SDK_SRC_UTILS_FILE_UTILS_H_

#include <filesystem>
#include <string>

#include "asr_sdk/status.h"

namespace asr_sdk::internal {

bool FileExists(const std::filesystem::path& path);
bool DirectoryExists(const std::filesystem::path& path);
Status ReadTextFile(const std::filesystem::path& path, std::string* out);
std::filesystem::path ResolveUnder(const std::filesystem::path& root,
                                   const std::string& path);

}  // namespace asr_sdk::internal

#endif  // ASR_SDK_SRC_UTILS_FILE_UTILS_H_
