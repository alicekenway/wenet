#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include "asr_sdk/decode_context.h"
#include "flashlight_decoder/word_spelling_generator.h"

using namespace asr_sdk::internal::flashlight_decoder;
void Expect(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
int main(int argc, char** argv) {
  try {
    GeneratedWordSpellings a{{{1}, {2, 3}, {4, 5, 6}}, {}};
    GeneratedWordSpellings b{{{7}, {8, 9}, {10, 11, 12}}, {}};
    GeneratedWordSpellings empty;
    auto two = CombineWordSpellings({&a, &b}, 100000, 64);
    Expect(two.ok() && two.value().size() == 9, "two-word product must have nine paths");
    auto three = CombineWordSpellings({&a, &b, &a}, 100000, 64);
    Expect(three.ok() && three.value().size() == 27, "three-word product must have 27 paths");
    Expect(!CombineWordSpellings({&a, &b, &a}, 16, 64).ok(), "explicit low limit must fail");
    Expect(!CombineWordSpellings({&a, &b}, 8, 64).ok(), "no partial path set on overflow");
    Expect(!CombineWordSpellings({&a, &b}, 100000, 5).ok(), "longest selected path must be checked");
    Expect(!CombineWordSpellings({&a, &empty}, 100000, 64).ok(), "empty component must fail");
    Expect(!CombineWordSpellings({}, 100000, 64).ok(), "empty name must fail");
    std::vector<const GeneratedWordSpellings*> huge(100, &a);
    Expect(!CombineWordSpellings(huge, std::numeric_limits<size_t>::max(), 1000).ok(),
           "product calculation must not overflow");
    Expect(asr_sdk::ContactListOptions{}.max_total_dynamic_forms == 100000,
           "default context budget changed");
    if (argc == 3) {
      asr_sdk::internal::sherpa_onnx_wenet::TokenTable tokens(argv[1]);
      const int blank = tokens.Contains("<blank>") ? tokens.Id("<blank>") : tokens.Id("<blk>");
      WordSpellingGenerator generator(tokens, blank, "▁", argv[2]);
      std::string word;
      while (std::getline(std::cin, word)) {
        const auto generated = generator.Generate(word);
        std::cout << word;
        for (const auto& path : generated.paths) {
          std::cout << '\t';
          for (size_t i = 0; i < path.size(); ++i) {
            if (i) std::cout << ' ';
            std::cout << path[i];
          }
        }
        std::cout << '\n';
      }
    } else {
      Expect(argc == 1, "usage: word_spelling_test [tokens.txt sentencepiece.model] < words");
      std::cout << "word spelling product tests passed\n";
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
