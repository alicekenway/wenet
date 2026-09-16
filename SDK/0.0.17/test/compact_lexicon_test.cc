#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <unistd.h>

#include "flashlight/lib/text/decoder/Trie.h"
#include "flashlight_decoder/compact_lexicon.h"
#include "flashlight_decoder/compact_lexicon_builder.h"

namespace {

namespace fs = std::filesystem;
using asr_sdk::internal::flashlight_decoder::CompactLexicon;
using asr_sdk::internal::flashlight_decoder::LexiconEntry;

void Expect(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<uint8_t> Read(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  return std::vector<uint8_t>(std::istreambuf_iterator<char>(input), {});
}

void Write(const fs::path& path, const std::vector<uint8_t>& bytes) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}

template <typename Function>
void ExpectFailure(Function function, const std::string& scenario) {
  try {
    function();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(scenario + " unexpectedly succeeded");
}

LexiconEntry Entry(int word, std::vector<int> tokens) {
  LexiconEntry entry;
  entry.word_id = word;
  entry.word = "word" + std::to_string(word);
  entry.token_ids = std::move(tokens);
  return entry;
}

}  // namespace

int main() {
  const fs::path directory = fs::temp_directory_path() /
      ("compact_lexicon_test_" + std::to_string(getpid()));
  try {
    fs::remove_all(directory);
    fs::create_directories(directory);
    const fs::path first_path = directory / "first.bin";
    const fs::path second_path = directory / "second.bin";

    auto trie = std::make_shared<fl::lib::text::Trie>(6, 1);
    const std::vector<LexiconEntry> entries{
        Entry(3, {1}), Entry(1, {1, 2}), Entry(2, {1, 2}),
        Entry(1, {1, 3}), Entry(4, {4, 5, 2})};
    const std::vector<float> scores{0.5f, 1.25f, 2.5f, -0.25f, 3.0f};
    for (size_t index = 0; index < entries.size(); ++index) {
      trie->insert(entries[index].token_ids, entries[index].word_id,
                   scores[index]);
    }
    trie->smear(fl::lib::text::SmearingMode::MAX);
    std::unordered_map<const fl::lib::text::TrieNode*, float> context;
    context[trie->search({1}).get()] = 9.0f;
    context[trie->search({1, 2}).get()] = 8.0f;

    const uint64_t dependency = 0x1020304050607080ULL;
    const auto first_stats =
        asr_sdk::internal::flashlight_decoder::WriteCompactLexicon(
            first_path, trie, context, entries, 6, 8, 1, 0, 0, dependency);
    const auto second_stats =
        asr_sdk::internal::flashlight_decoder::WriteCompactLexicon(
            second_path, trie, context, entries, 6, 8, 1, 0, 0, dependency);
    Expect(Read(first_path) == Read(second_path),
           "repeated compilation must be byte deterministic");
    Expect(first_stats.payload_hash == second_stats.payload_hash,
           "repeated compilation hash differs");

    asr_sdk::internal::flashlight_decoder::VerifyCompactLexiconAgainstLegacy(
        first_path, trie, context, entries, 6, 8, 1, 0, 0, dependency);
    const auto compact = CompactLexicon::Load(first_path, dependency);
    const auto prefix = compact->Child(CompactLexicon::kRoot, 1);
    Expect(prefix != CompactLexicon::kInvalidState,
           "prefix state is missing");
    const auto prefix_labels = compact->Labels(prefix);
    Expect(prefix_labels.size() == 1 && prefix_labels[0] == 3,
           "prefix word label was not preserved");
    const auto homophone = compact->Child(prefix, 2);
    const auto homophone_labels = compact->Labels(homophone);
    Expect(homophone_labels.size() == 2 && homophone_labels[0] == 1 &&
               homophone_labels[1] == 2,
           "homophone labels or their ordering were not preserved");
    Expect(compact->WordSpellings(1) ==
               std::vector<std::vector<int>>({{1, 2}, {1, 3}}),
           "multiple pronunciations were not preserved");
    Expect(compact->Lookahead(prefix, true) == 9.0f,
           "context lookahead was not preserved");
    Expect(compact->Child(prefix, 5) == CompactLexicon::kInvalidState,
           "absent edge was reported as present");

    ExpectFailure([&] { CompactLexicon::Load(first_path, dependency + 1); },
                  "dependency mismatch");
    std::vector<uint8_t> corrupt = Read(first_path);
    corrupt[0] ^= 0xff;
    Write(directory / "bad_magic.bin", corrupt);
    ExpectFailure([&] { CompactLexicon::Load(directory / "bad_magic.bin"); },
                  "bad magic");
    corrupt = Read(first_path);
    corrupt.back() ^= 0x01;
    Write(directory / "bad_payload.bin", corrupt);
    ExpectFailure(
        [&] { CompactLexicon::Load(directory / "bad_payload.bin"); },
        "bad payload checksum");
    corrupt = Read(first_path);
    corrupt.resize(corrupt.size() - 1);
    Write(directory / "truncated.bin", corrupt);
    ExpectFailure([&] { CompactLexicon::Load(directory / "truncated.bin"); },
                  "truncated binary");

    fs::remove_all(directory);
    return 0;
  } catch (const std::exception& error) {
    fs::remove_all(directory);
    std::cerr << error.what() << '\n';
    return 1;
  }
}
