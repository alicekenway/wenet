#ifndef WENET_SDK_SRC_UTILS_CHECKSUM_H_
#define WENET_SDK_SRC_UTILS_CHECKSUM_H_

#include <filesystem>
#include <string>

#include "utils/status.h"

namespace wenet_sdk::internal {

Status ValidateChecksumFileIfPresent(const std::filesystem::path& model_dir);
std::string Sha256HexForFile(const std::filesystem::path& path);
std::string HexDigestForFileForDiagnostics(const std::filesystem::path& path);

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_UTILS_CHECKSUM_H_
