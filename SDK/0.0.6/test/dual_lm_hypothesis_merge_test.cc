#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "flashlight_decoder/decoded_hypothesis.h"

namespace {

using asr_sdk::internal::flashlight_decoder::DecodedContactCandidate;
using asr_sdk::internal::flashlight_decoder::DecodedEntity;
using asr_sdk::internal::flashlight_decoder::DecodedHypothesis;
using asr_sdk::internal::flashlight_decoder::DecodedWord;
using asr_sdk::internal::flashlight_decoder::DecoderSource;
using asr_sdk::internal::flashlight_decoder::HasCompletedContact;
using asr_sdk::internal::flashlight_decoder::MergeMainAndEligibleContactHypotheses;

void Expect(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

DecodedHypothesis MainHyp(double score) {
  DecodedHypothesis hyp;
  hyp.total_score = score;
  hyp.source = DecoderSource::kMain;
  hyp.mapped_words = {{1, "please", 0, 1}, {2, "call", 1, 2}};
  return hyp;
}

DecodedHypothesis ContactHyp(double score, bool completed,
                             std::string id = "ada-42") {
  DecodedHypothesis hyp;
  hyp.total_score = score;
  hyp.source = DecoderSource::kContact;
  hyp.mapped_words = {{1, "please", 0, 1}, {2, "call", 1, 2}};
  if (completed) {
    DecodedEntity entity;
    entity.text = "Ada Wong";
    entity.score = -0.7f;
    entity.candidates.push_back(
        DecodedContactCandidate{std::move(id), "Ada Wong"});
    hyp.entities.push_back(std::move(entity));
  }
  return hyp;
}

}  // namespace

int main() {
  try {
    const auto main = MainHyp(-10.0);
    const auto unfinished_contact = ContactHyp(-1.0, false);
    Expect(!HasCompletedContact(unfinished_contact),
           "unfinished contact path must be ineligible");

    auto merged = MergeMainAndEligibleContactHypotheses(
        {main}, {unfinished_contact}, 5);
    Expect(merged.size() == 1, "unfinished contact must not enter merge");
    Expect(merged.front().source == DecoderSource::kMain,
           "main path should remain when no contact completed");

    const auto completed_contact = ContactHyp(-2.0, true);
    Expect(HasCompletedContact(completed_contact),
           "completed contact must be eligible");
    merged = MergeMainAndEligibleContactHypotheses(
        {main}, {completed_contact}, 5);
    Expect(merged.size() == 2,
           "entity-bearing contact and plain main paths are semantically distinct");
    Expect(merged.front().source == DecoderSource::kContact,
           "higher-scoring completed contact should win");

    const auto weaker_duplicate = ContactHyp(-3.0, true);
    merged = MergeMainAndEligibleContactHypotheses(
        {}, {completed_contact, weaker_duplicate}, 5);
    Expect(merged.size() == 1,
           "equivalent contact entities should deduplicate");
    Expect(merged.front().total_score == -2.0,
           "best duplicate contact path should survive");

    std::cout << "dual_lm_hypothesis_merge_test passed\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "dual_lm_hypothesis_merge_test failed: " << e.what() << "\n";
    return 1;
  }
}
