#include <iostream>
#include <string>

#include "model/model_metadata.h"

namespace {

std::string ArgValue(int argc, char** argv, const std::string& name,
                     const std::string& fallback = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) {
      return argv[i + 1];
    }
  }
  return fallback;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model_dir = ArgValue(argc, argv, "--model_dir");
  if (model_dir.empty()) {
    std::cerr << "usage: validate_model_package --model_dir MODEL_DIR\n";
    return 2;
  }

  wenet_sdk::internal::ModelMetadata metadata;
  auto status = wenet_sdk::internal::LoadModelMetadata(model_dir, &metadata);
  if (!status.ok()) {
    std::cerr << status.message() << "\n";
    return 1;
  }
  status = wenet_sdk::internal::ValidateModelPackageFiles(metadata, true);
  if (!status.ok()) {
    std::cerr << status.message() << "\n";
    return 1;
  }
  std::cout << "model package is valid\n";
  std::cout << wenet_sdk::internal::ModelSummary(metadata);
  return 0;
}
