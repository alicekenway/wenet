#include "flashlight_decoder/lexicon_loader.h"

#include <cctype>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

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

std::vector<std::string> SplitAsciiWhitespace(const std::string& text) {
  std::istringstream in(text);
  std::vector<std::string> items;
  std::string item;
  while (in >> item) {
    items.push_back(item);
  }
  return items;
}

std::string EntryKey(const std::string& word,
                     const std::vector<std::string>& pieces) {
  std::string key = word;
  key.push_back('\x1f');
  for (const std::string& piece : pieces) {
    key += piece;
    key.push_back('\x1f');
  }
  return key;
}

}  // namespace

std::vector<LexiconEntry> LoadLexicon(
    const std::filesystem::path& path, const WordDictionary& words,
    const sherpa_onnx_wenet::TokenTable& tokens) {
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("failed to open lexicon.txt: " + path.string());
  }

  std::vector<LexiconEntry> entries;
  std::unordered_set<std::string> seen;
  std::string line;
  int line_no = 0;
  while (std::getline(in, line)) {
    ++line_no;
    line = Trim(line);
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> fields = SplitAsciiWhitespace(line);
    if (fields.size() < 2) {
      throw std::runtime_error(path.string() + ":" +
                               std::to_string(line_no) +
                               ": lexicon line needs word and spelling");
    }
    const std::string word = fields.front();
    if (!words.Contains(word)) {
      throw std::runtime_error(path.string() + ":" +
                               std::to_string(line_no) +
                               ": lexicon word not in words.txt: " + word);
    }
    std::vector<std::string> spelling(fields.begin() + 1, fields.end());
    const std::string key = EntryKey(word, spelling);
    if (seen.find(key) != seen.end()) {
      continue;
    }
    seen.insert(key);

    LexiconEntry entry;
    entry.word = word;
    entry.word_id = words.Id(word);
    entry.tokens = std::move(spelling);
    entry.token_ids.reserve(entry.tokens.size());
    for (const std::string& token : entry.tokens) {
      if (!tokens.Contains(token)) {
        throw std::runtime_error(path.string() + ":" +
                                 std::to_string(line_no) +
                                 ": spelling token not in tokens.txt: " +
                                 token);
      }
      const int token_id = tokens.Id(token);
      if (token_id >= tokens.ModelVocabSize()) {
        throw std::runtime_error(path.string() + ":" +
                                 std::to_string(line_no) +
                                 ": spelling token is outside ONNX vocab: " +
                                 token);
      }
      entry.token_ids.push_back(token_id);
    }
    entries.push_back(std::move(entry));
  }
  if (entries.empty()) {
    throw std::runtime_error("lexicon.txt has no entries: " + path.string());
  }
  return entries;
}

}  // namespace asr_sdk::internal::flashlight_decoder
