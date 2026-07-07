#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "flashlight_decoder/decoded_hypothesis.h"
#include "flashlight_decoder/output_sequence_mapper.h"
#include "flashlight_decoder/word_dictionary.h"

namespace {

using asr_sdk::internal::flashlight_decoder::DecodedWord;
using asr_sdk::internal::flashlight_decoder::OutputSequenceMapper;
using asr_sdk::internal::flashlight_decoder::WordDictionary;
namespace fs = std::filesystem;

void WriteFile(const fs::path& path, const std::string& text) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write " + path.string());
  }
  out << text;
}

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Fn>
void ExpectThrows(Fn fn, const std::string& message) {
  try {
    fn();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error(message);
}

std::vector<int> Ids(const std::vector<DecodedWord>& words) {
  std::vector<int> out;
  for (const auto& word : words) {
    out.push_back(word.word_id);
  }
  return out;
}

DecodedWord Word(int id, std::string text, int frame) {
  DecodedWord word;
  word.word_id = id;
  word.text = std::move(text);
  word.start_frame = frame;
  word.end_frame = frame + 1;
  return word;
}

}  // namespace

int main() {
  try {
    const fs::path dir = fs::temp_directory_path() / "asr_sdk_mapper_test";
    fs::create_directories(dir);
    const fs::path words_path = dir / "words.txt";
    WriteFile(words_path,
              "<unk> 0\n"
              "A 1\n"
              "B 2\n"
              "C 3\n"
              "D 4\n"
              "X 5\n"
              "Y 6\n"
              "牛 7\n"
              "乃 8\n"
              "奶 9\n"
              "CIRCULATION 10\n");
    WordDictionary words(words_path);

    const fs::path empty_path = dir / "empty_mapping.txt";
    WriteFile(empty_path, "\n# comments only\n");
    auto empty = OutputSequenceMapper::Load(empty_path, words);
    Expect(empty.RuleCount() == 0, "empty mapping should have zero rules");
    Expect(empty.RewriteIds({1, 2, 3}) == std::vector<int>({1, 2, 3}),
           "empty mapping should be identity");

    const fs::path longest_path = dir / "longest_mapping.txt";
    WriteFile(longest_path,
              "A B -> X\n"
              "A B C -> Y\n"
              "牛 乃 -> 牛 奶\n");
    auto mapper = OutputSequenceMapper::Load(longest_path, words);
	    Expect(mapper.RuleCount() == 3, "expected three mapping rules");
	    Expect(mapper.RewriteIds({1, 2, 3, 4}) == std::vector<int>({6, 4}),
	           "longest mapping failed");
    Expect(mapper.RewriteIds({1, 2, 4, 1, 2}) ==
               std::vector<int>({5, 4, 5}),
           "multiple mapping replacements failed");
	    std::vector<DecodedWord> mapped =
	        mapper.RewriteWords({Word(7, "牛", 10), Word(8, "乃", 11)});
    Expect(Ids(mapped) == std::vector<int>({7, 9}), "UTF-8 mapping failed");
    Expect(mapped[0].text == "牛" && mapped[1].text == "奶",
           "UTF-8 mapped text failed");

    const fs::path duplicate_path = dir / "duplicate_mapping.txt";
    WriteFile(duplicate_path,
              "A B -> X\n"
              "A B -> Y\n");
    ExpectThrows([&]() { OutputSequenceMapper::Load(duplicate_path, words); },
                 "duplicate source should fail");

    const fs::path text_path = dir / "text_mapping.txt";
    WriteFile(text_path,
              "CIRCS -> CIRCULATION\n"
              "A -> OUTSIDE_VOCAB\n");
    auto text_mapper = OutputSequenceMapper::Load(text_path, words);
    Expect(text_mapper.RuleCount() == 2, "text mapping rule count failed");
    std::vector<DecodedWord> text_mapped =
        text_mapper.RewriteWords({Word(-1, "CIRCS", 20), Word(1, "A", 21)});
    Expect(Ids(text_mapped) == std::vector<int>({10, -1}),
           "text mapping ids failed");
    Expect(text_mapped[0].text == "CIRCULATION" &&
               text_mapped[1].text == "OUTSIDE_VOCAB",
           "text mapping output failed");

    const fs::path final_path = dir / "final_mapping.txt";
    WriteFile(final_path, "X D -> Y\n");
    auto final_mapper = OutputSequenceMapper::Load(final_path, words);
    std::vector<DecodedWord> raw = {Word(1, "A", 0), Word(2, "B", 1),
                                    Word(4, "D", 2), Word(1, "A", 3),
                                    Word(2, "B", 4)};
    std::vector<DecodedWord> am_mapped = mapper.RewriteWords(raw);
    std::vector<DecodedWord> final_mapped =
        final_mapper.RewriteWords(am_mapped);
    Expect(Ids(am_mapped) == std::vector<int>({5, 4, 5}),
           "AM-stage mapping failed");
    Expect(Ids(final_mapped) == std::vector<int>({6, 5}),
           "final-stage mapping failed");

	    std::cout << "output_sequence_mapper_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "output_sequence_mapper_test failed: " << e.what() << "\n";
    return 1;
  }
}
