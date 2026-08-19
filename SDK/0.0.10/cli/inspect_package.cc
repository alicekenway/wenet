#include <iostream>
#include <string>

#include "asr_sdk/config.h"
#include "package/model_package.h"
#include "package/model_package_validator.h"

namespace {

void Usage(const char* argv0) {
  std::cerr << "usage: " << argv0 << " --model_dir DIR\n";
}

}  // namespace

int main(int argc, char** argv) {
  std::string model_dir;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--model_dir" && i + 1 < argc) {
      model_dir = argv[++i];
    } else {
      Usage(argv[0]);
      return 2;
    }
  }
  if (model_dir.empty()) {
    Usage(argv[0]);
    return 2;
  }
  asr_sdk::EngineConfig config;
  config.model_dir = model_dir;
  auto package_or = asr_sdk::internal::LoadModelPackage(config);
  if (!package_or.ok()) {
    std::cerr << package_or.status().ToString() << "\n";
    return 1;
  }
  const auto report =
      asr_sdk::internal::InspectModelPackage(package_or.value());
  for (const auto& line : report.lines) {
    std::cout << line << "\n";
  }
  std::cout << "wenet_linkage: static\n";
  std::cout << "onnxruntime_linkage: dynamic\n";
  return report.ok ? 0 : 1;
}
