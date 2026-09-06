#include "flashlight_decoder/debug_trace.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

using asr_sdk::internal::flashlight_decoder::BuildDebugJson;
using asr_sdk::internal::flashlight_decoder::DecodedEntity;
using asr_sdk::internal::flashlight_decoder::DecodedHypothesis;
using asr_sdk::internal::flashlight_decoder::DecodedWord;
using asr_sdk::internal::flashlight_decoder::LmEventTrace;

void ExpectContains(const std::string& haystack, const std::string& needle,
                    const std::string& message) {
  if (haystack.find(needle) == std::string::npos) {
    throw std::runtime_error(message + " missing: " + needle);
  }
}

DecodedWord Word(int id, std::string text) {
  DecodedWord word;
  word.word_id = id;
  word.text = std::move(text);
  word.start_frame = id;
  word.end_frame = id + 1;
  return word;
}

}  // namespace

int main() {
  try {
    DecodedHypothesis hyp;
    hyp.first_pass_score = -11.0;
    hyp.am_score = -10.0;
    hyp.lm_score = -2.5;
    hyp.weighted_lm_score = -3.125;
    hyp.total_score = -12.5;
    hyp.raw_words = {Word(1, "raw")};
    hyp.am_mapped_words = {Word(2, "am"), Word(3, "mapped")};
    DecodedWord contact_first = Word(5, "Ada");
    contact_first.is_contact = true;
    contact_first.contact_entity_index = 0;
    DecodedWord contact_second = Word(6, "Wong");
    contact_second.is_contact = true;
    contact_second.contact_entity_index = 0;
    hyp.mapped_words = {Word(4, "final"), contact_first, contact_second};
    DecodedEntity entity;
    entity.type = "address";
    entity.slot_token = "<ADDRESS>";
    entity.text = "Ada Wong";
    entity.score = 3.5f;
    hyp.entities.push_back(entity);
    hyp.lm_events.push_back(
        LmEventTrace{"final", "general.bin", "ngram", -1.0, -3.0,
                     2.0, 1.0});

    const std::string json =
        BuildDebugJson({"stream_init", "final_nbest count=1"}, "", {hyp},
                       true);
    ExpectContains(json, "\"mode\":\"max_fusion\"", "debug mode");
    ExpectContains(json, "\"final_nbest\"", "final nbest");
    ExpectContains(json, "\"text\":\"final Ada Wong\"", "final text");
    ExpectContains(json, "\"debug_text\":\"final Ada_Wong<ADDRESS>\"",
                   "tagged generic-slot debug text");
    ExpectContains(json, "\"bias_score\":3.5", "generic bias score");
    ExpectContains(json, "\"raw_text\":\"raw\"", "raw text");
    ExpectContains(json, "\"am_mapped_text\":\"am mapped\"",
                   "AM-mapped text");
    ExpectContains(json, "\"decoder_source\":\"max_fusion\"",
                   "max-fusion decoder source");
    ExpectContains(json, "\"filename\":\"general.bin\"",
                   "winning LM filename");
    ExpectContains(json, "\"reference_score\":-3", "UNK reference score");
    ExpectContains(json, "\"am_score\":-10", "AM score");
    ExpectContains(json, "\"lm_score\":-2.5", "LM score");
    ExpectContains(json, "\"weighted_lm_score\":-3.125",
                   "weighted LM score");
    ExpectContains(json, "\"total_score\":-12.5", "total score");

    std::cout << "debug_trace_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "debug_trace_test failed: " << e.what() << "\n";
    return 1;
  }
}
