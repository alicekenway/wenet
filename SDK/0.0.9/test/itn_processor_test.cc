#include <iostream>
#include <memory>
#include <string>

#include "itn/itn_processor.h"

namespace {

bool Expect(bool value, const std::string& message) {
  if (!value) std::cerr << "FAIL: " << message << "\n";
  return value;
}

class SnapshotStream final : public asr_sdk::AsrStream {
 public:
  explicit SnapshotStream(asr_sdk::AsrResult result) : result_(std::move(result)) {}
  asr_sdk::Status AcceptPcm16(const int16_t*, size_t, int) override { return asr_sdk::Status::Ok(); }
  bool DecodeReady() const override { return false; }
  asr_sdk::Status Decode() override {
    if (result_.text == "one hundred") result_.text = "one hundred and one";
    return asr_sdk::Status::Ok();
  }
  asr_sdk::AsrResult GetResult() const override { return result_; }
  asr_sdk::AsrResult GetFinalResult() override { result_.is_final = true; return result_; }
  asr_sdk::Status SetInputFinished() override { return asr_sdk::Status::Ok(); }
  asr_sdk::Status Reset() override { return asr_sdk::Status::Ok(); }
 private:
  asr_sdk::AsrResult result_;
};

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  auto processor_or = asr_sdk::internal::CreateItnProcessor(argv[1], argv[2]);
  if (!Expect(processor_or.ok(), processor_or.status().ToString())) return 1;
  auto processor = std::move(processor_or).value();
  bool ok = true;
  ok &= Expect(processor->Normalize("F M ONE HUNDRED POINT NINE") == "FM100.9",
               "radio normalization");
  ok &= Expect(processor->Normalize("M H THREE SEVENTY") == "MH370",
               "identifier normalization");
  ok &= Expect(processor->Normalize("Call seven one one five four") == "Call 71154",
               "context number and case restoration");
  ok &= Expect(processor->Normalize("one hundred") == "100", "partial 100");
  ok &= Expect(processor->Normalize("one hundred and one") == "101",
               "later replacement 101");

  asr_sdk::AsrResult raw;
  raw.text = "Call seven one one five four";
  raw.nbest.push_back({raw.text, 3.5f, {}, {}});
  auto stream = asr_sdk::internal::WrapWithItn(
      std::unique_ptr<asr_sdk::AsrStream>(new SnapshotStream(raw)), processor);
  const asr_sdk::AsrResult result = stream->GetResult();
  ok &= Expect(result.text == "Call 71154", "best result normalized");
  ok &= Expect(result.raw_text == raw.text, "raw best retained");
  ok &= Expect(result.nbest.size() == 1 && result.nbest[0].text == raw.text &&
                   result.nbest[0].score == 3.5f,
               "n-best and score remain in decoder space");

  asr_sdk::AsrResult partial;
  partial.text = "one hundred";
  auto revising = asr_sdk::internal::WrapWithItn(
      std::unique_ptr<asr_sdk::AsrStream>(new SnapshotStream(partial)), processor);
  ok &= Expect(revising->GetResult().text == "100", "first partial snapshot");
  ok &= Expect(revising->Decode().ok(), "decode later partial snapshot");
  const auto later = revising->GetResult();
  ok &= Expect(later.text == "101" && later.raw_text == "one hundred and one",
               "later partial replaces earlier normalized snapshot");
  return ok ? 0 : 1;
}
