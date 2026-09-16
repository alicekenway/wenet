#include "flashlight_decoder/word_spelling_generator.h"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <utility>

namespace asr_sdk::internal::flashlight_decoder {
namespace {
struct FoldEntry { uint32_t code; const char* folded; };
constexpr FoldEntry kFoldTable[] = {
#include "flashlight_decoder/unicode_casefold.inc"
};

// Work on Unicode scalars, not bytes; reject invalid and overlong UTF-8.
std::vector<size_t> Offsets(const std::string& text,
                            std::string* folded = nullptr) {
  std::vector<size_t> offsets;
  size_t pos = 0;
  while (pos < text.size()) {
    offsets.push_back(pos);
    const auto lead = static_cast<unsigned char>(text[pos]);
    size_t length = lead < 0x80 ? 1 : lead >= 0xc2 && lead <= 0xdf ? 2 :
                    lead >= 0xe0 && lead <= 0xef ? 3 :
                    lead >= 0xf0 && lead <= 0xf4 ? 4 : 0;
    if (!length || pos + length > text.size()) {
      throw std::invalid_argument("invalid UTF-8 in contact word");
    }
    uint32_t code = lead & (length == 1 ? 0x7f : (1 << (7-length))-1);
    for (size_t i = 1; i < length; ++i) {
      const auto byte = static_cast<unsigned char>(text[pos+i]);
      if ((byte & 0xc0) != 0x80) {
        throw std::invalid_argument("invalid UTF-8 continuation in contact word");
      }
      code = (code << 6) | (byte & 0x3f);
    }
    if ((length == 2 && code < 0x80) || (length == 3 && code < 0x800) ||
        (length == 4 && code < 0x10000) || code > 0x10ffff ||
        (code >= 0xd800 && code <= 0xdfff)) {
      throw std::invalid_argument("invalid UTF-8 scalar in contact word");
    }
    if (folded) {
      const auto* end = std::end(kFoldTable);
      const auto* found = std::lower_bound(std::begin(kFoldTable), end, code,
          [](const FoldEntry& item, uint32_t value) { return item.code < value; });
      if (found != end && found->code == code) *folded += found->folded;
      else folded->append(text, pos, length);
    }
    pos += length;
  }
  offsets.push_back(text.size());
  return offsets;
}

std::string Fold(const std::string& text) {
  std::string out;
  Offsets(text, &out);
  return out;
}
void AddUnique(std::vector<int> path, TokenPaths* paths) {
  if (!path.empty() && std::find(paths->begin(), paths->end(), path) == paths->end()) {
    paths->push_back(std::move(path));
  }
}
}  // namespace

WordSpellingGenerator::WordSpellingGenerator(
    const sherpa_onnx_wenet::TokenTable& tokens, int blank_id,
    std::string boundary, const std::filesystem::path& model)
    : boundary_(std::move(boundary)) {
  if (model.empty()) {
    throw std::invalid_argument(
        "automatic DCC requires sentencepiece_model in sdk_model.json; "
        "explicit am_tokens remain supported without it");
  }
  const auto loaded = processor_.Load(model.string());
  if (!loaded.ok()) throw std::invalid_argument("cannot load sentencepiece_model: " + loaded.ToString());
  int count = 0, missing = 0;
  for (int i = 0; i < processor_.GetPieceSize(); ++i) {
    const auto& piece = processor_.IdToPiece(i);
    if (piece == "<s>" || piece == "</s>") continue;
    ++count;
    if (!tokens.Contains(piece) || tokens.Id(piece) >= tokens.ModelVocabSize()) {
      ++missing;
      diagnostics_.push_back("tokenizer piece unavailable in AM: " + piece);
    }
  }
  if (count && static_cast<double>(count-missing)/count < 0.99) {
    throw std::invalid_argument("sentencepiece_model/token table coverage is below 99%; mismatched tokenizer");
  }
  for (int id = 0; id < tokens.ModelVocabSize(); ++id) {
    const auto& token = tokens.Token(id);
    if (id == blank_id || token.empty() || token.front() == '#' ||
        token == boundary_ || (token.front() == '<' && token.back() == '>')) continue;
    exact_.emplace(token, id);
    const auto folded = Fold(token);
    const auto found = folded_.find(folded);
    if (found == folded_.end() || id < found->second) folded_[folded] = id;
    max_piece_characters_ = std::max(max_piece_characters_,
        std::max(Offsets(token).size()-1, Offsets(folded).size()-1));
  }
  if (exact_.empty()) throw std::invalid_argument("no usable BPE tokens for automatic DCC");
}

int WordSpellingGenerator::Resolve(const std::string& surface) const {
  const auto exact = exact_.find(surface);
  if (exact != exact_.end()) return exact->second;
  const auto folded = folded_.find(Fold(surface));
  return folded == folded_.end() ? -1 : folded->second;
}

GeneratedWordSpellings WordSpellingGenerator::Generate(const std::string& word) const {
  GeneratedWordSpellings out;
  const auto chars = Offsets(word);
  if (word.empty()) return out;
  std::vector<std::string> pieces;
  const auto encoded = processor_.Encode(word, &pieces);
  std::vector<int> canonical;
  bool valid = encoded.ok() && !pieces.empty();
  for (const auto& piece : pieces) {
    const auto found = exact_.find(piece);
    if (found == exact_.end()) { valid = false; break; }
    canonical.push_back(found->second);
  }
  if (valid) AddUnique(std::move(canonical), &out.paths);
  else out.diagnostics.push_back("canonical unavailable: " + word);

  const std::string text = boundary_ + word;
  const auto offsets = Offsets(text);
  std::vector<std::optional<std::vector<int>>> best(offsets.size());
  best[0] = std::vector<int>();
  for (size_t pos = 0; pos+1 < offsets.size(); ++pos) {
    if (!best[pos]) continue;
    const size_t end = std::min(offsets.size()-1, pos+max_piece_characters_);
    for (size_t next = pos+1; next <= end; ++next) {
      const int id = Resolve(text.substr(offsets[pos], offsets[next]-offsets[pos]));
      if (id < 0) continue;
      auto candidate = *best[pos]; candidate.push_back(id);
      if (!best[next] || candidate.size() < best[next]->size() ||
          (candidate.size() == best[next]->size() && candidate < *best[next])) {
        best[next] = std::move(candidate);
      }
    }
  }
  if (best.back()) AddUnique(std::move(*best.back()), &out.paths);
  else out.diagnostics.push_back("shortest unavailable: " + word);

  std::vector<int> characters;
  for (size_t i = 0; i+1 < chars.size(); ++i) {
    const auto surface = (i == 0 ? boundary_ : "") +
                         word.substr(chars[i], chars[i+1]-chars[i]);
    const int id = Resolve(surface);
    if (id < 0) { characters.clear(); break; }
    characters.push_back(id);
  }
  if (!characters.empty()) AddUnique(std::move(characters), &out.paths);
  else out.diagnostics.push_back("characters unavailable: " + word);
  return out;
}

StatusOr<TokenPaths> CombineWordSpellings(
    const std::vector<const GeneratedWordSpellings*>& words,
    size_t max_paths, size_t max_tokens) {
  if (words.empty()) return Status::InvalidArgument("spoken form is empty");
  size_t count = 1, longest = 0;
  for (const auto* word : words) {
    if (word->paths.empty()) return Status::InvalidArgument("contact word has no valid token path");
    if (count > max_paths / word->paths.size()) {
      return Status::FailedPrecondition("contact combinations exceed path budget; no paths were truncated");
    }
    count *= word->paths.size();
    size_t length = 0;
    for (const auto& path : word->paths) length = std::max(length, path.size());
    if (length > max_tokens || longest > max_tokens-length) {
      return Status::FailedPrecondition("contact combination exceeds max_tokens_per_spoken_form");
    }
    longest += length;
  }
  TokenPaths paths(1);
  for (const auto* word : words) {
    TokenPaths next;
    next.reserve(paths.size()*word->paths.size());
    for (const auto& prefix : paths) {
      for (const auto& spelling : word->paths) {
        auto candidate = prefix;
        candidate.insert(candidate.end(), spelling.begin(), spelling.end());
        next.push_back(std::move(candidate));
      }
    }
    paths = std::move(next);
  }
  std::sort(paths.begin(), paths.end());
  paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
  return paths;
}
}  // namespace asr_sdk::internal::flashlight_decoder
