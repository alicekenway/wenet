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

#include "asr_sdk/decode_context.h"
#include "flashlight_decoder/compiled_decode_context.h"
#include "flashlight_decoder/flashlight_decoder_resource.h"
#include "flashlight_decoder/max_fusion_lm.h"

namespace {
namespace fs = std::filesystem;
using namespace asr_sdk::internal;
using namespace asr_sdk::internal::flashlight_decoder;

void Write(const fs::path& path, const std::string& text) {
  std::ofstream out(path); out << text;
}
void Expect(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}
void Near(float a, float b, const std::string& message) {
  if (std::fabs(a-b) > 1e-4f) throw std::runtime_error(
      message + " actual=" + std::to_string(a) + " expected=" + std::to_string(b));
}
std::string Ngram(float please, float call) {
  return "\\data\\\nngram 1=7\nngram 2=4\n\n\\1-grams:\n"
         "-1\t<unk>\t0\n-1\t<s>\t0\n-1\t</s>\t0\n-0.4\tplease\t0\n"
         "-0.4\tcall\t0\n-0.5\t<CONTACT>\t0\n"
         "-0.5\t<ADDRESS>\t0\n\n\\2-grams:\n" +
         std::to_string(please) + "\t<s> please\n" +
         std::to_string(call) +
         "\tplease call\n-0.2\t<CONTACT> please\n"
         "-0.3\t<ADDRESS> please\n\n\\end\\\n";
}
std::string Bias(const std::string& slot, float terminal_score) {
  return "\\data\\\n"
         "ngram 1=7\n"
         "ngram 2=4\n\n"
         "\\1-grams:\n"
         "0\t<unk>\t0\n"
         "0\t<s>\t0\n"
         "0\t</s>\t0\n"
         "0\tplease\t0\n"
         "0\tcall\t0\n"
         "0\t<CONTACT>\t0\n"
         "0\t<ADDRESS>\t0\n\n"
         "\\2-grams:\n"
         "0\t<s> please\n"
         "0\tplease call\n" + std::to_string(terminal_score) +
         "\tcall " + slot + "\n-5\t<s> " + slot + "\n\n\\end\\\n";
}
}  // namespace

int main() {
  const fs::path dir = fs::temp_directory_path() /
      ("asr_sdk_max_fusion_test_" + std::to_string(getpid()));
  try {
    fs::remove_all(dir); fs::create_directories(dir);
    Write(dir/"tokens.txt", "<blk> 0\n▁ 1\nP 2\nC 3\nA 4\nD 5\n#0 6\n");
    Write(dir/"words.txt", "<unk> 0\nplease 1\ncall 2\n<CONTACT> 3\n"
                           "<ADDRESS> 4\nother 5\n");
    Write(dir/"lexicon.txt", "please ▁ P\ncall ▁ C\n");
    Write(dir/"one.arpa", Ngram(-0.1f, -0.8f));
    Write(dir/"two.arpa", Ngram(-0.6f, -0.1f));
    Write(dir/"bias.arpa", Bias("<CONTACT>", -9.0f));
    Write(dir/"address.arpa", Bias("<ADDRESS>", -4.0f));
    FixedLmConfig one{"one.bin", dir/"one.arpa", FixedLmType::kNgram,
                      1.0, true, 0.0, 0.5};
    FixedLmConfig two{"two.bin", dir/"two.arpa", FixedLmType::kNgram,
                      1.0, true, 0.0, 2.0};
    FixedLmConfig bias;
    bias.filename="bias.bin"; bias.path=dir/"bias.arpa";
    bias.type=FixedLmType::kBias; bias.weight=2.0;
    bias.accumulation_factor=0.5; bias.slots={"<CONTACT>"};
    FixedLmConfig address;
    address.filename="address.bin"; address.path=dir/"address.arpa";
    address.type=FixedLmType::kBias; address.weight=3.0;
    address.accumulation_factor=0.0; address.slots={"<ADDRESS>"};
    FlashlightDecoderOptions options; options.smearing="max";
    auto base=std::make_shared<FlashlightDecoderResource>(
        dir/"tokens.txt",dir/"words.txt",dir/"lexicon.txt",
        std::vector<FixedLmConfig>{one,two,bias,address},fs::path(),fs::path(),options,
        "<blk>","▁","<unk>",0.5);
    auto state=base->WordLm()->start(false);
    auto please=base->WordLm()->score(state,1); state=please.first;
    auto call=base->WordLm()->score(state,2);
    Near(please.second,0.5f,
         "LM1 wins first word after its relative bonus is clipped");
    Near(call.second,0.9f,"LM2 wins second word by relative bonus");
    auto other=base->WordLm()->score(call.first,5);
    Near(other.second,0.0f,
         "OOV word has zero relative bonus because raw equals UNK reference");

    asr_sdk::DecodeContextConfig config;
    asr_sdk::SlotClass slot; slot.slot_token="<CONTACT>";
    asr_sdk::SlotValueEntry value; value.value_id="ada"; value.display_name="Ada";
    value.spoken_forms.push_back({"Ada",{"▁","A","D"},2});
    slot.values.push_back(value); config.slots.push_back(slot);
    asr_sdk::SlotClass address_slot; address_slot.slot_token="<ADDRESS>";
    asr_sdk::SlotValueEntry address_value;
    address_value.value_id="ada-address";
    address_value.display_name="Ada Address";
    address_value.spoken_forms.push_back({"Ada",{"▁","A","D"},2});
    address_slot.values.push_back(address_value);
    config.slots.push_back(address_slot);
    auto compiled=CompiledDecodeContext::Compile(
        base,config,std::make_shared<ContextOwnerToken>());
    Expect(compiled.ok(),compiled.status().ToString());
    auto overlay=compiled.value()->SlotDecoderResource();
    auto base_lm=std::dynamic_pointer_cast<MaxFusionLm>(base->WordLm());
    auto overlay_lm=std::dynamic_pointer_cast<MaxFusionLm>(overlay->WordLm());
    Expect(base_lm!=nullptr && overlay_lm!=nullptr,
           "base and DCC resources must both use MaxFusionLm");
    Expect(base_lm->FixedResources()==overlay_lm->FixedResources(),
           "DCC resource must share the already-loaded fixed LM resources");
    Expect(base->LexiconTrie()==overlay->LexiconTrie(),
           "DCC resource must share the immutable base trie");
    Expect(&base->AmTokens()==&overlay->AmTokens() &&
               &base->OutputWords()==&overlay->OutputWords() &&
               &base->WordSpellings()==&overlay->WordSpellings(),
           "DCC resource must share package-sized dictionaries and spellings");
    Expect(overlay->OverlayTrie()!=nullptr,
           "DCC resource must own a separate contact overlay trie");
    Expect(base_lm->Models().size()==overlay_lm->Models().size(),
           "DCC resource must expose the same fixed LM set");
    for (std::size_t i=0;i<base_lm->Models().size();++i) {
      Expect(base_lm->Models()[i].lm==overlay_lm->Models()[i].lm,
             "DCC resource must not reload a fixed KenLM model");
    }
    const auto& forms=compiled.value()->DynamicContacts()->Forms();
    Expect(forms.size()==2,
           "identical AM spelling in two slot classes must remain distinct");
    const auto contact_it=std::find_if(forms.begin(),forms.end(),[](const auto& f){
      return f.slot_token=="<CONTACT>";
    });
    const auto address_it=std::find_if(forms.begin(),forms.end(),[](const auto& f){
      return f.slot_token=="<ADDRESS>";
    });
    Expect(contact_it!=forms.end() && address_it!=forms.end(),
           "both configured generic slot classes must compile");
    const auto& form=*contact_it;
    Expect(overlay->OverlayTrie()->search(form.token_ids)!=nullptr,
           "contact spelling must be present in the overlay trie");
    auto s=overlay->WordLm()->start(false);
    s=overlay->WordLm()->score(s,1).first;
    s=overlay->WordLm()->score(s,2).first;
    auto dynamic=overlay->WordLm()->score(s,form.dynamic_word_id);
    Near(dynamic.second,2.0f*9.0f*1.5f-0.5f,
         "slot gets only weighted geometric bias and virtual-word correction");
    auto after=overlay->WordLm()->score(dynamic.first,1);
    Near(after.second,0.8f,
         "ngrams advance through <CONTACT> and score following word");
    auto a=overlay->WordLm()->start(false);
    a=overlay->WordLm()->score(a,1).first;
    a=overlay->WordLm()->score(a,2).first;
    auto dynamic_address=overlay->WordLm()->score(a,address_it->dynamic_word_id);
    Near(dynamic_address.second,3.0f*4.0f-0.5f,
         "second bias LM independently scores its declared generic slot");
    auto after_address=overlay->WordLm()->score(dynamic_address.first,1);
    Near(after_address.second,0.7f,
         "ngrams advance through <ADDRESS> before the following word");
    fs::remove_all(dir);
    return 0;
  } catch (const std::exception& e) {
    fs::remove_all(dir); std::cerr << e.what() << "\n"; return 1;
  }
}
