#include "sherpa_onnx_wenet/token_table.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <stdexcept>

namespace asr_sdk::internal::sherpa_onnx_wenet {
namespace {

std::string TrimRight(std::string s) {
  while (!s.empty() &&
         (s.back() == '\n' || s.back() == '\r' ||
          std::isspace(static_cast<unsigned char>(s.back())))) {
    s.pop_back();
  }
  return s;
}

bool IsByteToken(const std::string& token, int* value) {
  if (token.size() != 6 || token.rfind("<0x", 0) != 0 || token.back() != '>') {
    return false;
  }
  try {
    *value = std::stoi(token.substr(3, 2), nullptr, 16);
    return *value >= 0 && *value <= 255;
  } catch (...) {
    return false;
  }
}

}  // namespace

TokenTable::TokenTable(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open tokens.txt: " + path.string());
  }

  std::string line;
  while (std::getline(in, line)) {
    line = TrimRight(line);
    if (line.empty()) {
      continue;
    }
    const size_t last_space = line.find_last_of(" \t");
    if (last_space == std::string::npos) {
      throw std::runtime_error("invalid tokens.txt line: " + line);
    }
    const int id = std::stoi(line.substr(last_space + 1));
    std::string token = TrimRight(line.substr(0, last_space));
    if (id < 0) {
      throw std::runtime_error("negative token id in tokens.txt");
    }
    if (id >= static_cast<int>(id_to_token_.size())) {
      id_to_token_.resize(static_cast<size_t>(id + 1));
    }
    id_to_token_[static_cast<size_t>(id)] = token;
    if (!token.empty() && token[0] == '#' && model_vocab_size_ == 0) {
      model_vocab_size_ = id;
    }
    if (token == "<blk>" || token == "<blank>") {
      blank_id_ = id;
    }
  }
  if (id_to_token_.empty()) {
    throw std::runtime_error("tokens.txt is empty");
  }
  if (model_vocab_size_ == 0) {
    model_vocab_size_ = static_cast<int>(id_to_token_.size());
  }
}

const std::string& TokenTable::Token(int id) const {
  if (id < 0 || id >= static_cast<int>(id_to_token_.size())) {
    throw std::runtime_error("token id out of range: " + std::to_string(id));
  }
  return id_to_token_[static_cast<size_t>(id)];
}

std::string TokenTable::DecodeIds(const std::vector<int>& ids) const {
  std::string out;
  std::string byte_buffer;
  auto flush_bytes = [&]() {
    if (!byte_buffer.empty()) {
      out += byte_buffer;
      byte_buffer.clear();
    }
  };

  for (int id : ids) {
    if (id == blank_id_) {
      continue;
    }
    const std::string& token = Token(id);
    int byte_value = 0;
    if (IsByteToken(token, &byte_value)) {
      byte_buffer.push_back(static_cast<char>(byte_value));
      continue;
    }
    flush_bytes();
    if (token.empty() || token[0] == '<' || token[0] == '#') {
      continue;
    }
    std::string normalized = token;
    const std::string space_marker = "▁";
    size_t pos = 0;
    while ((pos = normalized.find(space_marker, pos)) != std::string::npos) {
      normalized.replace(pos, space_marker.size(), " ");
      ++pos;
    }
    out += normalized;
  }
  flush_bytes();
  while (!out.empty() && out.front() == ' ') {
    out.erase(out.begin());
  }
  while (!out.empty() && out.back() == ' ') {
    out.pop_back();
  }
  return out;
}

}  // namespace asr_sdk::internal::sherpa_onnx_wenet
