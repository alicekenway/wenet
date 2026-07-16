#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unistd.h>
#include <utility>
#include <vector>

#include "asr_sdk/decode_context.h"
#include "flashlight_decoder/compiled_decode_context.h"
#include "flashlight_decoder/flashlight_decoder_resource.h"
#include "flashlight_decoder/flashlight_result_mapper.h"

namespace {

namespace fs = std::filesystem;
using asr_sdk::ContactEntry;
using asr_sdk::ContactSpokenForm;
using asr_sdk::DecodeContextConfig;
using asr_sdk::internal::flashlight_decoder::CompiledDecodeContext;
using asr_sdk::internal::flashlight_decoder::ContextOwnerToken;
using asr_sdk::internal::flashlight_decoder::ConvertFlashlightResult;
using asr_sdk::internal::flashlight_decoder::DecoderSource;
using asr_sdk::internal::flashlight_decoder::FlashlightDecoderOptions;
using asr_sdk::internal::flashlight_decoder::FlashlightDecoderResource;
using asr_sdk::internal::flashlight_decoder::RuntimeContactForm;

void WriteFile(const fs::path& path, const std::string& content) {
  std::ofstream out(path);
  if (!out) {
    throw std::runtime_error("failed to write " + path.string());
  }
  out << content;
}

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void ExpectNear(float actual, float expected, const std::string& message) {
  if (std::fabs(actual - expected) > 1e-5f) {
    throw std::runtime_error(message + " actual=" + std::to_string(actual) +
                             " expected=" + std::to_string(expected));
  }
}

std::string MainArpa() {
  return R"arpa(\data\
ngram 1=5
ngram 2=2

\1-grams:
-1.0	<unk>	0
-1.0	<s>	0
-1.0	</s>	0
-0.2	please	0
-0.3	call	0

\2-grams:
-0.1	<s> please
-0.1	please call

\end\
)arpa";
}

std::string ContactBiasArpa() {
  return R"arpa(\data\
ngram 1=6
ngram 2=4

\1-grams:
0	<unk>	0
0	<s>	0
0	</s>	0
0	please	0
0	call	0
0	<CONTACT>	0

\2-grams:
0	<s> please
0	please call
-9	call <CONTACT>
-5	<s> <CONTACT>

\end\
)arpa";
}

const RuntimeContactForm& FormWithWordCount(
    const std::vector<RuntimeContactForm>& forms, int count) {
  for (const auto& form : forms) {
    if (form.logical_word_count == count) {
      return form;
    }
  }
  throw std::runtime_error("missing requested contact form");
}

}  // namespace

int main() {
  const fs::path dir = fs::temp_directory_path() /
                       ("asr_sdk_contact_class_lm_test_" +
                        std::to_string(getpid()));
  try {
    fs::remove_all(dir);
    fs::create_directories(dir);
    const fs::path tokens = dir / "tokens.txt";
    const fs::path words = dir / "words.txt";
    const fs::path lexicon = dir / "lexicon.txt";
    const fs::path main_lm = dir / "main.arpa";
    const fs::path contact_lm = dir / "contact.arpa";

    WriteFile(tokens,
              "<blk> 0\n"
              "▁ 1\n"
              "P 2\n"
              "C 3\n"
              "A 4\n"
              "D 5\n"
              "B 6\n"
              "O 7\n"
              "#0 8\n");
    WriteFile(words,
              "<unk> 0\n"
              "please 1\n"
              "call 2\n"
              "other 3\n"
              "<CONTACT> 4\n");
    WriteFile(lexicon, "please ▁ P\ncall ▁ C\nother ▁ O\n");
    // The main LM intentionally lacks both `other` and <CONTACT>.  Missing
    // entries must be zero-score/reset, rather than <unk> or rejection.
    WriteFile(main_lm, MainArpa());
    WriteFile(contact_lm, ContactBiasArpa());

    FlashlightDecoderOptions options;
    options.lm_weight = 1.25;
    options.word_score = -0.5;
    options.smearing = "max";
    auto base = std::make_shared<FlashlightDecoderResource>(
        tokens, words, lexicon, main_lm, fs::path(), options, "<blk>", "▁",
        "<unk>", "<CONTACT>", contact_lm, /*contact_lm_weight=*/2.0,
        /*contact_lm_accumulation_factor=*/0.5,
        /*contact_lm_max_bonus=*/9.0);
    Expect(base->SupportsRuntimeContacts(), "contact package capability");

    DecodeContextConfig config;
    config.contacts = {
        ContactEntry{"ada", "Ada Wong", {ContactSpokenForm{
                                           "Ada Wong", {"▁", "A", "▁", "D"}, 2}}},
        ContactEntry{"bob", "Bob", {ContactSpokenForm{
                                       "Bob", {"▁", "B", "▁", "O"}, 1}}},
    };
    const auto owner = std::make_shared<ContextOwnerToken>();
    auto context_or = CompiledDecodeContext::Compile(base, config, owner);
    Expect(context_or.ok(), context_or.status().ToString());
    const auto context = std::move(context_or).value();
    const auto& overlay = context->ContactDecoderResource();
    Expect(overlay != nullptr, "contact overlay resource");
    ExpectNear(static_cast<float>(overlay->Options().lm_weight), 1.0f,
               "overlay uses pre-weighted composite scores");

    const auto& forms = context->DynamicContacts()->Forms();
    const RuntimeContactForm& two_word_form = FormWithWordCount(forms, 2);
    const RuntimeContactForm& one_word_form = FormWithWordCount(forms, 1);

    const auto contact_node =
        overlay->LexiconTrie()->search(two_word_form.token_ids);
    Expect(contact_node != nullptr, "contact form is present in combined trie");
    const auto contact_label = std::find(contact_node->labels.begin(),
                                         contact_node->labels.end(),
                                         two_word_form.dynamic_word_id);
    Expect(contact_label != contact_node->labels.end(),
           "combined trie has the dynamic contact label");
    const size_t contact_index =
        static_cast<size_t>(contact_label - contact_node->labels.begin());
    ExpectNear(contact_node->scores[contact_index], 0.0f,
               "dynamic contact trie lookahead is neutral");

    const auto wrapper = overlay->WordLm();
    auto wrapper_state = wrapper->start(false);
    auto main_state = base->WordLm()->start(false);
    for (int word_id : {1, 2}) {
      float raw_main = 0.0f;
      std::tie(main_state, raw_main) = base->WordLm()->score(main_state, word_id);
      const auto scored = wrapper->score(wrapper_state, word_id);
      wrapper_state = scored.first;
      ExpectNear(scored.second, raw_main * 1.25f,
                 "ordinary contact-path word uses weighted main LM");
    }

    const auto ada = wrapper->score(wrapper_state, two_word_form.dynamic_word_id);
    const float expected_ada = 2.0f * 9.0f * 1.5f - 0.5f;
    ExpectNear(ada.second, expected_ada,
               "two-word contact uses geometric bonus plus word correction");

    auto standalone_state = wrapper->start(false);
    const auto bob =
        wrapper->score(standalone_state, one_word_form.dynamic_word_id);
    ExpectNear(bob.second, 2.0f * 5.0f,
               "standalone rule uses its own encoded bonus");
    const auto after_standalone = wrapper->score(bob.first, 2);
    Expect(std::isinf(after_standalone.second) && after_standalone.second < 0.0f,
           "standalone contact rejects trailing words");

    auto unmatched_state = wrapper->start(false);
    unmatched_state = wrapper->score(unmatched_state, 1).first;
    const auto unmatched =
        wrapper->score(unmatched_state, one_word_form.dynamic_word_id);
    Expect(std::isinf(unmatched.second) && unmatched.second < 0.0f,
           "unconfigured contact context is rejected");

    auto oov_state = wrapper->start(false);
    const auto oov = wrapper->score(oov_state, 3);
    ExpectNear(oov.second, 0.0f, "missing word has zero composite LM score");
    auto raw_oov_state = base->WordLm()->start(false);
    const auto raw_oov = base->WordLm()->score(raw_oov_state, 3);
    ExpectNear(raw_oov.second, 0.0f,
               "main KenLM missing word is zero-score/reset too");
    auto after_oov = wrapper->score(oov.first, 2);
    const auto after_oov_contact =
        wrapper->score(after_oov.first, one_word_form.dynamic_word_id);
    ExpectNear(after_oov_contact.second, 2.0f * 9.0f,
               "pattern history restarts after a missing word");

    fl::lib::text::DecodeResult result(3);
    result.words = {1, 2, two_word_form.dynamic_word_id};
    auto mapped = ConvertFlashlightResult(result, *overlay);
    Expect(mapped.source == DecoderSource::kContact,
           "contact overlay result source");
    Expect(mapped.entities.size() == 1, "one mapped contact entity");
    ExpectNear(mapped.entities.front().score, 2.0f * 9.0f * 1.5f,
               "entity score reports weighted geometric contact bonus");

    fs::remove_all(dir);
    std::cout << "contact_class_lm_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(dir);
    std::cerr << "contact_class_lm_test failed: " << e.what() << "\n";
    return 1;
  }
}
