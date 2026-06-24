#include "utils/file_utils.h"

#include <fstream>
#include <sstream>

namespace asr_sdk::internal {

bool FileExists(const std::filesystem::path& path) {
  std::error_code ec;
  return std::filesystem::is_regular_file(path, ec);
}

bool DirectoryExists(const std::filesystem::path& path) {
  std::error_code ec;
  return std::filesystem::is_directory(path, ec);
}

Status ReadTextFile(const std::filesystem::path& path, std::string* out) {
  if (out == nullptr) {
    return Status::InvalidArgument("output pointer is null");
  }
  std::ifstream in(path);
  if (!in) {
    return Status::NotFound("failed to open text file: " + path.string());
  }
  std::ostringstream buffer;
  buffer << in.rdbuf();
  *out = buffer.str();
  return Status::Ok();
}

std::filesystem::path ResolveUnder(const std::filesystem::path& root,
                                   const std::string& path) {
  std::filesystem::path p(path);
  if (p.is_absolute()) {
    return p;
  }
  return root / p;
}

}  // namespace asr_sdk::internal
