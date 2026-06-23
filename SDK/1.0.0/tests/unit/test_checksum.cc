#include <cassert>
#include <filesystem>
#include <fstream>

#include "utils/checksum.h"

int main() {
  const auto dir = std::filesystem::temp_directory_path() /
                   "wenet_sdk_checksum_test";
  std::filesystem::create_directories(dir);
  {
    std::ofstream out(dir / "data.txt", std::ios::binary);
    out << "abc";
  }
  assert(wenet_sdk::internal::Sha256HexForFile(dir / "data.txt") ==
         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  {
    std::ofstream out(dir / "checksum.sha256");
    out << "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        << "  data.txt\n";
  }
  auto status = wenet_sdk::internal::ValidateChecksumFileIfPresent(dir);
  assert(status.ok());
  std::filesystem::remove_all(dir);
  return 0;
}
