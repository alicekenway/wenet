#include "flashlight_decoder/word_dictionary.h"

#include <cctype>
#include <fstream>
#include <stdexcept>

namespace asr_sdk::internal::flashlight_decoder {
namespace {

std::string Trim(std::string text) {
  size_t begin = 0;
  while (begin < text.size() &&
         std::isspace(static_cast<unsigned char>(text[begin]))) {
    ++begin;
  }
  size_t end = text.size();
  while (end > begin &&
         std::isspace(static_cast<unsigned char>(text[end - 1]))) {
    --end;
  }
  return text.substr(begin, end - begin);
}

}  // namespace

WordDictionary::WordDictionary(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open words.txt: " + path.string());
  }

  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    line = Trim(line);
    if (line.empty()) {
      continue;
    }
    const size_t last_space = line.find_last_of(" \t");
    if (last_space == std::string::npos) {
      throw std::runtime_error(path.string() + ":" +
                               std::to_string(line_no) +
                               ": expected '<word> <id>'");
    }
    std::string word = Trim(line.substr(0, last_space));
    const std::string id_text = Trim(line.substr(last_space + 1));
    if (word.empty() || id_text.empty()) {
      throw std::runtime_error(path.string() + ":" +
                               std::to_string(line_no) +
                               ": empty word or id");
    }
    int id = -1;
    try {
      id = std::stoi(id_text);
    } catch (...) {
      throw std::runtime_error(path.string() + ":" +
                               std::to_string(line_no) +
                               ": invalid word id '" + id_text + "'");
    }
    if (id < 0) {
      throw std::runtime_error(path.string() + ":" +
                               std::to_string(line_no) +
                               ": negative word id");
    }
    if (word_to_id_.find(word) != word_to_id_.end()) {
      throw std::runtime_error(path.string() + ":" +
                               std::to_string(line_no) +
                               ": duplicate word '" + word + "'");
    }
    if (id >= static_cast<int>(id_to_word_.size())) {
      id_to_word_.resize(static_cast<size_t>(id + 1));
    }
    if (!id_to_word_[static_cast<size_t>(id)].empty()) {
      throw std::runtime_error(path.string() + ":" +
                               std::to_string(line_no) +
                               ": duplicate word id " + std::to_string(id));
    }
    id_to_word_[static_cast<size_t>(id)] = word;
    word_to_id_[word] = id;
  }

  if (id_to_word_.empty()) {
    throw std::runtime_error("words.txt is empty: " + path.string());
  }
  for (int id = 0; id < static_cast<int>(id_to_word_.size()); ++id) {
    if (id_to_word_[static_cast<size_t>(id)].empty()) {
      throw std::runtime_error(path.string() +
                               ": sparse word id, missing id " +
                               std::to_string(id));
    }
  }
  if (!Contains("<unk>")) {
    throw std::runtime_error(path.string() + ": missing required <unk> word");
  }
}

bool WordDictionary::Contains(const std::string& word) const {
  return word_to_id_.find(word) != word_to_id_.end();
}

bool WordDictionary::ContainsId(int id) const {
  return id >= 0 && id < static_cast<int>(id_to_word_.size());
}

int WordDictionary::Id(const std::string& word) const {
  const auto it = word_to_id_.find(word);
  if (it == word_to_id_.end()) {
    throw std::runtime_error("word not in words.txt: " + word);
  }
  return it->second;
}

const std::string& WordDictionary::Word(int id) const {
  if (!ContainsId(id)) {
    throw std::runtime_error("word id out of range: " + std::to_string(id));
  }
  return id_to_word_[static_cast<size_t>(id)];
}

}  // namespace asr_sdk::internal::flashlight_decoder
