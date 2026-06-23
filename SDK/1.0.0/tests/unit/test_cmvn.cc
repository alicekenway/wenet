#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <vector>

#include "frontend/cmvn.h"

namespace {

bool Near(float a, float b) {
  return std::fabs(a - b) < 1.0e-5f;
}

void WriteText(const std::filesystem::path& path, const std::string& text) {
  std::ofstream out(path);
  out << text;
}

}  // namespace

int main() {
  const auto dir = std::filesystem::temp_directory_path() /
                   "wenet_sdk_cmvn_test";
  std::filesystem::create_directories(dir);

  const auto json_path = dir / "global_cmvn_json";
  WriteText(json_path,
            "{\"mean_stat\":[2.0,4.0],\"var_stat\":[5.0,20.0],"
            "\"frame_num\":2}");
  wenet_sdk::internal::Cmvn cmvn;
  auto status = cmvn.LoadIfPresent(json_path, 2);
  assert(status.ok());
  assert(cmvn.enabled());
  std::vector<float> frame = {2.0f, 5.0f};
  cmvn.Apply(&frame);
  assert(Near(frame[0], 1.0f / std::sqrt(1.5f)));
  assert(Near(frame[1], 3.0f / std::sqrt(6.0f)));

  status = cmvn.LoadIfPresent(json_path, 3);
  assert(!status.ok());

  const auto kaldi_path = dir / "global_cmvn_kaldi";
  WriteText(kaldi_path, "[ 2.0 4.0 2.0 5.0 20.0 0 ]");
  status = cmvn.LoadIfPresent(kaldi_path, 2);
  assert(status.ok());
  frame = {2.0f, 5.0f};
  cmvn.Apply(&frame);
  assert(Near(frame[0], 1.0f / std::sqrt(1.5f)));
  assert(Near(frame[1], 3.0f / std::sqrt(6.0f)));

  const auto flat_path = dir / "global_cmvn_flat";
  WriteText(flat_path, "1.0 2.0 0.5 0.25");
  status = cmvn.LoadIfPresent(flat_path, 2);
  assert(status.ok());
  frame = {3.0f, 6.0f};
  cmvn.Apply(&frame);
  assert(Near(frame[0], 1.0f));
  assert(Near(frame[1], 1.0f));

  std::filesystem::remove_all(dir);
  return 0;
}
