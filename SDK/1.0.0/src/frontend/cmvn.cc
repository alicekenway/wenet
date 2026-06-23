#include "frontend/cmvn.h"

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>

namespace wenet_sdk::internal {
namespace {

constexpr double kVarianceFloor = 1.0e-20;

bool ParseDoubleToken(const std::string& token, double* value) {
  if (value == nullptr || token.empty()) {
    return false;
  }
  char* end = nullptr;
  errno = 0;
  const double parsed = std::strtod(token.c_str(), &end);
  if (end == token.c_str() || *end != '\0' || errno == ERANGE) {
    return false;
  }
  *value = parsed;
  return true;
}

Status ParseJsonNumber(const std::string& text, const std::string& key,
                       double* value) {
  const std::string needle = "\"" + key + "\"";
  const size_t key_pos = text.find(needle);
  if (key_pos == std::string::npos) {
    return Status::InvalidArgument("missing CMVN JSON key: " + key);
  }
  const size_t colon = text.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return Status::InvalidArgument("invalid CMVN JSON key: " + key);
  }
  size_t pos = text.find_first_not_of(" \t\r\n", colon + 1);
  if (pos == std::string::npos) {
    return Status::InvalidArgument("invalid CMVN JSON number: " + key);
  }
  char* end = nullptr;
  errno = 0;
  const double parsed = std::strtod(text.c_str() + pos, &end);
  if (end == text.c_str() + pos || errno == ERANGE) {
    return Status::InvalidArgument("invalid CMVN JSON number: " + key);
  }
  *value = parsed;
  return Status::OK();
}

Status ParseJsonArray(const std::string& text, const std::string& key,
                      std::vector<double>* values) {
  if (values == nullptr) {
    return Status::InvalidArgument("null CMVN JSON array output");
  }
  values->clear();
  const std::string needle = "\"" + key + "\"";
  const size_t key_pos = text.find(needle);
  if (key_pos == std::string::npos) {
    return Status::InvalidArgument("missing CMVN JSON key: " + key);
  }
  const size_t colon = text.find(':', key_pos + needle.size());
  const size_t open = text.find('[', colon == std::string::npos ? key_pos : colon);
  if (colon == std::string::npos || open == std::string::npos) {
    return Status::InvalidArgument("invalid CMVN JSON array: " + key);
  }
  const size_t close = text.find(']', open + 1);
  if (close == std::string::npos) {
    return Status::InvalidArgument("unterminated CMVN JSON array: " + key);
  }

  const char* ptr = text.c_str() + open + 1;
  const char* end_array = text.c_str() + close;
  while (ptr < end_array) {
    while (ptr < end_array &&
           (*ptr == ',' || std::isspace(static_cast<unsigned char>(*ptr)))) {
      ++ptr;
    }
    if (ptr >= end_array) {
      break;
    }
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(ptr, &end);
    if (end == ptr || errno == ERANGE || end > end_array) {
      return Status::InvalidArgument("invalid CMVN JSON array value: " + key);
    }
    values->push_back(parsed);
    ptr = end;
  }
  if (values->empty()) {
    return Status::InvalidArgument("empty CMVN JSON array: " + key);
  }
  return Status::OK();
}

Status SetFromStats(const std::vector<double>& mean_stat,
                    const std::vector<double>& var_stat, double frame_num,
                    int feature_dim, const std::filesystem::path& path,
                    std::vector<float>* mean,
                    std::vector<float>* inv_std) {
  if (feature_dim <= 0) {
    return Status::InvalidArgument("invalid CMVN feature dimension");
  }
  if (frame_num <= 0.0) {
    return Status::InvalidArgument("invalid CMVN frame count in " +
                                   path.string());
  }
  if (static_cast<int>(mean_stat.size()) != feature_dim ||
      static_cast<int>(var_stat.size()) != feature_dim) {
    return Status::InvalidArgument("CMVN dimension mismatch in " +
                                   path.string());
  }

  mean->resize(static_cast<size_t>(feature_dim));
  inv_std->resize(static_cast<size_t>(feature_dim));
  for (int i = 0; i < feature_dim; ++i) {
    const double m = mean_stat[static_cast<size_t>(i)] / frame_num;
    double variance = var_stat[static_cast<size_t>(i)] / frame_num - m * m;
    if (variance < kVarianceFloor) {
      variance = kVarianceFloor;
    }
    (*mean)[static_cast<size_t>(i)] = static_cast<float>(m);
    (*inv_std)[static_cast<size_t>(i)] =
        static_cast<float>(1.0 / std::sqrt(variance));
  }
  return Status::OK();
}

Status LoadJsonCmvn(const std::string& text, int feature_dim,
                    const std::filesystem::path& path,
                    std::vector<float>* mean,
                    std::vector<float>* inv_std) {
  std::vector<double> mean_stat;
  std::vector<double> var_stat;
  double frame_num = 0.0;
  auto status = ParseJsonArray(text, "mean_stat", &mean_stat);
  if (!status.ok()) {
    return status;
  }
  status = ParseJsonArray(text, "var_stat", &var_stat);
  if (!status.ok()) {
    return status;
  }
  status = ParseJsonNumber(text, "frame_num", &frame_num);
  if (!status.ok()) {
    return status;
  }
  return SetFromStats(mean_stat, var_stat, frame_num, feature_dim, path, mean,
                      inv_std);
}

Status LoadKaldiCmvn(const std::string& text, int feature_dim,
                     const std::filesystem::path& path,
                     std::vector<float>* mean,
                     std::vector<float>* inv_std) {
  std::istringstream in(text);
  std::vector<std::string> tokens;
  std::string token;
  while (in >> token) {
    tokens.push_back(token);
  }
  if (tokens.size() < 4 || tokens.front() != "[" || tokens.back() != "]") {
    return Status::InvalidArgument("invalid Kaldi CMVN file: " +
                                   path.string());
  }
  if (tokens[tokens.size() - 2] != "0") {
    return Status::InvalidArgument("invalid Kaldi CMVN trailer in " +
                                   path.string());
  }
  const size_t payload = tokens.size() - 4;
  if (payload % 2 != 0) {
    return Status::InvalidArgument("invalid Kaldi CMVN dimensions in " +
                                   path.string());
  }
  const int parsed_dim = static_cast<int>(payload / 2);
  if (parsed_dim != feature_dim) {
    return Status::InvalidArgument("CMVN dimension mismatch in " +
                                   path.string());
  }

  std::vector<double> mean_stat(static_cast<size_t>(feature_dim));
  std::vector<double> var_stat(static_cast<size_t>(feature_dim));
  for (int i = 0; i < feature_dim; ++i) {
    if (!ParseDoubleToken(tokens[static_cast<size_t>(1 + i)],
                          &mean_stat[static_cast<size_t>(i)])) {
      return Status::InvalidArgument("invalid Kaldi CMVN mean in " +
                                     path.string());
    }
  }
  double frame_num = 0.0;
  if (!ParseDoubleToken(tokens[static_cast<size_t>(feature_dim + 1)],
                        &frame_num)) {
    return Status::InvalidArgument("invalid Kaldi CMVN frame count in " +
                                   path.string());
  }
  for (int i = 0; i < feature_dim; ++i) {
    if (!ParseDoubleToken(
            tokens[static_cast<size_t>(feature_dim + 2 + i)],
            &var_stat[static_cast<size_t>(i)])) {
      return Status::InvalidArgument("invalid Kaldi CMVN variance in " +
                                     path.string());
    }
  }
  return SetFromStats(mean_stat, var_stat, frame_num, feature_dim, path, mean,
                      inv_std);
}

Status LoadFlatCmvn(const std::string& text, int feature_dim,
                    const std::filesystem::path& path,
                    std::vector<float>* mean,
                    std::vector<float>* inv_std) {
  std::istringstream in(text);
  std::vector<float> values;
  float value = 0.0f;
  while (in >> value) {
    values.push_back(value);
  }
  if (values.empty()) {
    return Status::InvalidArgument("CMVN file is empty: " + path.string());
  }
  if (static_cast<int>(values.size()) == feature_dim) {
    *mean = values;
    inv_std->assign(static_cast<size_t>(feature_dim), 1.0f);
    return Status::OK();
  }
  if (static_cast<int>(values.size()) == feature_dim * 2) {
    mean->assign(values.begin(), values.begin() + feature_dim);
    inv_std->assign(values.begin() + feature_dim, values.end());
    return Status::OK();
  }
  return Status::InvalidArgument("CMVN dimension mismatch in " +
                                 path.string());
}

}  // namespace

Status Cmvn::LoadIfPresent(const std::filesystem::path& path,
                           int feature_dim) {
  mean_.clear();
  inv_std_.clear();
  if (path.empty() || !std::filesystem::exists(path)) {
    return Status::OK();
  }

  std::ifstream in(path);
  if (!in) {
    return Status::NotFound("failed to open CMVN file: " + path.string());
  }

  const std::string text((std::istreambuf_iterator<char>(in)),
                         std::istreambuf_iterator<char>());
  const size_t first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return Status::InvalidArgument("CMVN file is empty: " + path.string());
  }

  if (text[first] == '{') {
    return LoadJsonCmvn(text, feature_dim, path, &mean_, &inv_std_);
  }
  if (text[first] == '[') {
    return LoadKaldiCmvn(text, feature_dim, path, &mean_, &inv_std_);
  }
  return LoadFlatCmvn(text, feature_dim, path, &mean_, &inv_std_);
}

void Cmvn::Apply(std::vector<float>* frame) const {
  if (frame == nullptr || mean_.empty()) {
    return;
  }
  const size_t n = std::min(frame->size(), mean_.size());
  for (size_t i = 0; i < n; ++i) {
    (*frame)[i] = ((*frame)[i] - mean_[i]) * inv_std_[i];
  }
}

}  // namespace wenet_sdk::internal
