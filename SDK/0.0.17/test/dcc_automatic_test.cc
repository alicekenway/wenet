#include <algorithm>
#include <chrono>
#include <cctype>
#include <fstream>
#include <future>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/resource.h>

#include "asr_sdk/asr_engine.h"
#include "flashlight_decoder/compiled_decode_context.h"
#include "flashlight_decoder/word_spelling_generator.h"
#include "package/model_package.h"

using namespace asr_sdk;
using namespace asr_sdk::internal;
using namespace asr_sdk::internal::flashlight_decoder;
using Clock = std::chrono::steady_clock;
void Expect(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}
double Milliseconds(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now()-start).count();
}
long RssKb() {
  std::ifstream input("/proc/self/status");
  std::string key, rest; long kb;
  while (input >> key) {
    if (key == "VmRSS:") { input >> kb; return kb; }
    std::getline(input, rest);
  }
  return -1;
}
size_t Nodes(const std::shared_ptr<fl::lib::text::Trie>& trie) {
  std::vector<const fl::lib::text::TrieNode*> pending{trie->getRoot()};
  size_t count = 0;
  while (!pending.empty()) {
    const auto* node = pending.back(); pending.pop_back(); ++count;
    for (const auto& child : node->children) pending.push_back(child.second.get());
  }
  return count;
}
ContactEntry Entry(std::string id, const std::string& text) {
  return {std::move(id), text, {{text, {}, 0}}};
}
int main(int argc, char** argv) {
  try {
    Expect(argc == 3 || argc == 4, "usage: dcc_automatic_test package contact_names [scale_count]");
    EngineConfig engine_config; engine_config.model_dir = argv[1];
    auto package_or = LoadModelPackage(engine_config);
    Expect(package_or.ok(), package_or.status().ToString());
    const auto& p = package_or.value();
    auto make_base = [&](const std::filesystem::path& model) {
      return std::make_shared<FlashlightDecoderResource>(p.tokens_txt, p.words_txt,
          p.lexicon_bin, p.fixed_lms, p.output_mapping_txt, p.final_output_mapping_txt,
          p.flashlight_options, p.blank_token, p.sil_token, p.unk_word,
          p.length_penalty, true, ComputeCompactLexiconDependencyHash(p), model);
    };
    const auto base_start = Clock::now();
    const auto base = make_base(p.sentencepiece_model);
    std::cout << "base_resource_ms=" << Milliseconds(base_start) << '\n';
    Expect(base->Lexicon() && !base->LexiconTrie(), "runtime must have only compact base lexicon");
    const auto owner = std::make_shared<ContextOwnerToken>();
    auto compile = [&](const DecodeContextConfig& config) {
      return CompiledDecodeContext::Compile(base, config, owner);
    };
    DecodeContextConfig config;
    std::ifstream names(argv[2]); Expect(names.good(), "cannot open contacts");
    std::string name;
    while (std::getline(names, name)) {
      if (name.empty()) continue;
      for (char& c : name) if (c >= 'a' && c <= 'z') c -= 'a'-'A';
      config.contacts.push_back(Entry(std::to_string(config.contacts.size()), name));
    }
    Expect(!config.contacts.empty(), "empty contact fixture");
    if (argc == 4) {
      const int count = std::stoi(argv[3]);
      Expect(count == 1000 || count == 2000, "scale must be 1000 or 2000");
      std::vector<std::string> words;
      for (int i = 0; i < base->OutputWords().Size() && words.size() < 13; ++i) {
        const auto& word = base->OutputWords().Word(i);
        if (word.empty() || word.front() == '<') continue;
        const auto generated = base->ContactWordSpeller().Generate(word);
        if (generated.paths.size() == 3 && std::all_of(generated.paths.begin(),
            generated.paths.end(), [](const auto& path) { return path.size() <= 20; })) {
          words.push_back(word);
        }
      }
      Expect(words.size() == 13, "need 13 three-path words for scale test");
      config.contacts.clear();
      for (int i = 0; i < count; ++i) {
        config.contacts.push_back(Entry(std::to_string(i),
            words[i/169] + " " + words[(i/13)%13] + " " + words[i%13]));
      }
    }
    const long before_rss = RssKb();
    const auto first_start = Clock::now();
    auto first = compile(config);
    Expect(first.ok(), first.status().ToString());
    const double first_ms = Milliseconds(first_start);
    const long after_rss = RssKb();
    const auto slot = first.value()->SlotDecoderResource();
    Expect(slot->Lexicon() == base->Lexicon(), "DCC copied the compact main lexicon");
    Expect(&slot->AmTokens() == &base->AmTokens() &&
           &slot->OutputWords() == &base->OutputWords(), "DCC copied vocabulary");
    Expect(&slot->ContactWordSpeller() == &base->ContactWordSpeller(), "tokenizer was not shared");
    Expect(base->OverlayTrie() && Nodes(base->OverlayTrie()) == 1 &&
           slot->OverlayTrie() && base->OverlayTrie() != slot->OverlayTrie(),
           "base empty overlay and contact overlay must remain separate");
    const auto& forms = first.value()->DynamicContacts()->Forms();
    const size_t original_form_count = forms.size();
    const std::string original_fingerprint = first.value()->Fingerprint();
    std::set<std::string> accepted, unique_words;
    for (const auto& form : forms) {
      auto node = slot->OverlayTrie()->search(form.token_ids);
      Expect(node && std::find(node->labels.begin(), node->labels.end(), form.dynamic_word_id)
          != node->labels.end(), "dynamic path or terminal label missing in trie");
      for (const auto& candidate : form.candidates) accepted.insert(candidate.contact_id);
    }
    Expect(accepted.size() == config.contacts.size(), "a contact was lost");
    for (const auto& contact : config.contacts) {
      std::istringstream input(contact.spoken_forms[0].text);
      std::string word; size_t expected = 1; int word_count = 0;
      while (input >> word) {
        expected *= base->ContactWordSpeller().Generate(word).paths.size();
        unique_words.insert(word); ++word_count;
      }
      size_t actual = 0;
      for (const auto& form : forms) {
        for (const auto& candidate : form.candidates) {
          if (candidate.contact_id == contact.contact_id) {
            ++actual;
            Expect(form.logical_word_count == word_count, "logical word count changed");
          }
        }
      }
      Expect(actual == expected, "full word-path product was not retained");
    }
    const auto generated_words = std::count_if(first.value()->Diagnostics().begin(),
        first.value()->Diagnostics().end(), [](const auto& d) { return d.rfind("word ", 0) == 0; });
    Expect(static_cast<size_t>(generated_words) == unique_words.size(), "words generated more than once per context");
    const auto second_start = Clock::now();
    auto second = compile(config);
    Expect(second.ok(), second.status().ToString());
    const double second_ms = Milliseconds(second_start);
    Expect(first.value()->Fingerprint() == second.value()->Fingerprint(), "nondeterministic context");
    std::cout << "contacts=" << accepted.size() << " forms=" << forms.size()
              << " unique_words=" << unique_words.size()
              << " overlay_nodes=" << Nodes(slot->OverlayTrie())
              << " first_compile_ms=" << first_ms << " repeat_compile_ms=" << second_ms
              << " rss_before_kb=" << before_rss << " rss_after_kb=" << after_rss << '\n';
    if (argc == 4) {
      Expect(forms.size() == config.contacts.size()*27, "scale did not retain every combination");
      std::cout << "scale test passed\n"; return 0;
    }
    // Concurrent first-time initialization of a fresh shared tokenizer.
    const auto fresh = make_base(p.sentencepiece_model);
    auto task = [&]() { return CompiledDecodeContext::Compile(fresh, config, owner); };
    auto worker = std::async(std::launch::async, task);
    const auto concurrent = task();
    const auto concurrent2 = worker.get();
    Expect(concurrent.ok() && concurrent2.ok(), "concurrent compilation failed");
    Expect(concurrent.value()->Fingerprint() == concurrent2.value()->Fingerprint(), "concurrent paths differ");
    auto limited = config; limited.contact_list.max_total_dynamic_forms = 1;
    limited.contact_list.skip_unencodable_forms = true;
    Expect(!compile(limited).ok(), "resource overflow must fail even with skip enabled");
    limited = config; limited.contact_list.max_token_segmentations_per_form = 1;
    Expect(!compile(limited).ok(), "per-form limit ignored");
    limited = config; limited.contact_list.max_tokens_per_spoken_form = 1;
    Expect(!compile(limited).ok(), "token length limit ignored");
    DecodeContextConfig invalid;
    invalid.contacts.push_back(Entry("invalid", "😀"));
    Expect(!compile(invalid).ok(), "unencodable name accepted");
    invalid.contact_list.skip_unencodable_forms = true;
    invalid.contacts.push_back(config.contacts[0]);
    auto skipped = compile(invalid);
    Expect(skipped.ok(), "explicit skip policy failed");
    Expect(std::any_of(skipped.value()->Diagnostics().begin(), skipped.value()->Diagnostics().end(),
        [](const auto& d) { return d.rfind("skipped ", 0) == 0; }), "missing skip diagnostic");
    auto no_model = make_base({});
    Expect(!CompiledDecodeContext::Compile(no_model, config, owner).ok(), "missing model must fail automatic DCC");
    Expect(CompiledDecodeContext::Compile(no_model, {}, owner).ok(), "empty DCC must not need model");
    DecodeContextConfig explicit_config;
    auto explicit_contact = config.contacts[0];
    const auto& source_form = forms.front();
    for (int id : source_form.token_ids) explicit_contact.spoken_forms[0].am_tokens.push_back(base->AmTokens().Token(id));
    explicit_config.contacts.push_back(explicit_contact);
    Expect(CompiledDecodeContext::Compile(no_model, explicit_config, owner).ok(), "explicit tokens need no model");
    explicit_config.contacts[0].spoken_forms[0].am_tokens = {p.blank_token};
    Expect(!CompiledDecodeContext::Compile(no_model, explicit_config, owner).ok(), "blank token accepted");
    const auto bad_model = make_base(p.tokens_txt);
    Expect(!CompiledDecodeContext::Compile(bad_model, config, owner).ok(), "corrupt model accepted");
    auto duplicate = config;
    auto alias = config.contacts[0]; alias.contact_id = "alias";
    duplicate.contacts.push_back(alias);
    auto duplicated = compile(duplicate);
    Expect(duplicated.ok(), "ambiguous contacts failed");
    Expect(std::any_of(duplicated.value()->DynamicContacts()->Forms().begin(),
        duplicated.value()->DynamicContacts()->Forms().end(),
        [](const auto& f) { return f.candidates.size() == 2; }), "ambiguous contact ID lost");
    Expect(first.value()->DynamicContacts()->Forms().size() == original_form_count &&
           first.value()->Fingerprint() == original_fingerprint &&
           Nodes(base->OverlayTrie()) == 1, "old context or base overlay changed");
    // Public engine/stream lifetime smoke test, using automatic text contacts.
    auto engine_or = AsrEngine::Create(engine_config);
    Expect(engine_or.ok(), engine_or.status().ToString());
    auto context = engine_or.value()->CompileDecodeContext(config);
    Expect(context.ok(), context.status().ToString());
    const auto stream_start = Clock::now();
    auto stream_worker = std::async(std::launch::async, [&]() {
      return engine_or.value()->CreateStream(context.value());
    });
    auto stream2 = engine_or.value()->CreateStream(context.value());
    auto stream1 = stream_worker.get();
    Expect(stream1.ok() && stream2.ok(), "shared-context streams failed");
    std::cout << "two_stream_create_ms=" << Milliseconds(stream_start) << '\n';
    std::cout << "automatic DCC integration tests passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n'; return 1;
  }
}
