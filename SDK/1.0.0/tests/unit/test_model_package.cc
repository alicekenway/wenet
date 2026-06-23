#include <cassert>
#include <filesystem>
#include <fstream>

#include "model/model_metadata.h"

int main() {
  wenet_sdk::internal::ModelMetadata metadata;
  auto status =
      wenet_sdk::internal::LoadModelMetadata("model_example", &metadata);
  assert(status.ok());
  assert(metadata.sample_rate == 16000);
  assert(metadata.streaming.chunk_size == 4);
  status = wenet_sdk::internal::ValidateModelPackageFiles(metadata, true);
  assert(status.ok());

  const auto dir = std::filesystem::temp_directory_path() /
                   "wenet_sdk_model_package_test";
  std::filesystem::remove_all(dir);
  std::filesystem::copy("model_example", dir,
                        std::filesystem::copy_options::recursive);

  status = wenet_sdk::internal::LoadModelMetadata(dir, &metadata);
  assert(status.ok());
  status = wenet_sdk::internal::ValidateModelPackageFiles(metadata, true);
  assert(status.ok());

  {
    std::ofstream out(dir / "tokens.txt");
    out << "HELLO 1\nWORLD 2\n";
  }
  status = wenet_sdk::internal::ValidateModelPackageFiles(metadata, true);
  assert(!status.ok());

  {
    std::ofstream out(dir / "tokens.txt");
    out << "<blank> 0\nHELLO 1\nWORLD 2\nSDK 3\n";
  }
  {
    std::ofstream out(dir / "global_cmvn");
    out << "1.0 2.0\n";
  }
  status = wenet_sdk::internal::ValidateModelPackageFiles(metadata, true);
  assert(!status.ok());

  std::filesystem::remove_all(dir);
  return 0;
}
