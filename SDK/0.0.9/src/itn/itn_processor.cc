#include "itn/itn_processor.h"

#include <algorithm>
#include <cctype>
#include <exception>
#include <utility>

#include "processor/wetext_processor.h"

namespace asr_sdk::internal {
namespace {

std::string LowerAscii(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
  return text;
}

std::string RestoreUnchangedWordCase(const std::string& raw,
                                     std::string normalized) {
  const std::string raw_lower = LowerAscii(raw);
  size_t raw_cursor = 0;
  for (size_t begin = 0; begin < normalized.size();) {
    if (!std::isalpha(static_cast<unsigned char>(normalized[begin]))) {
      ++begin;
      continue;
    }
    size_t end = begin + 1;
    while (end < normalized.size() &&
           std::isalpha(static_cast<unsigned char>(normalized[end]))) ++end;
    const std::string word = normalized.substr(begin, end - begin);
    const size_t found = raw_lower.find(LowerAscii(word), raw_cursor);
    if (found != std::string::npos) {
      normalized.replace(begin, end - begin, raw.substr(found, end - begin));
      raw_cursor = found + end - begin;
    }
    begin = end;
  }
  return normalized;
}

AsrResult Apply(const std::shared_ptr<const ItnProcessor>& processor,
                AsrResult result) {
  result.raw_text = result.text;
  if (result.text.empty()) return result;
  try {
    const std::string normalized = processor->Normalize(result.text);
    if (!normalized.empty()) result.text = normalized;
  } catch (const std::exception&) {
    // A partial hypothesis can temporarily have no valid grammar path. Keep
    // ASR usable and let the next complete replacement snapshot retry ITN.
  }
  return result;
}

class ItnAsrStream final : public AsrStream {
 public:
  ItnAsrStream(std::unique_ptr<AsrStream> inner,
               std::shared_ptr<const ItnProcessor> processor)
      : inner_(std::move(inner)), processor_(std::move(processor)) {}
  Status AcceptPcm16(const int16_t* s, size_t n, int rate) override {
    return inner_->AcceptPcm16(s, n, rate);
  }
  bool DecodeReady() const override { return inner_->DecodeReady(); }
  Status Decode() override { return inner_->Decode(); }
  AsrResult GetResult() const override { return Apply(processor_, inner_->GetResult()); }
  AsrResult GetFinalResult() override { return Apply(processor_, inner_->GetFinalResult()); }
  Status SetInputFinished() override { return inner_->SetInputFinished(); }
  Status Reset() override { return inner_->Reset(); }
 private:
  std::unique_ptr<AsrStream> inner_;
  std::shared_ptr<const ItnProcessor> processor_;
};

}  // namespace

ItnProcessor::ItnProcessor(const std::string& tagger,
                           const std::string& verbalizer)
    : processor_(new wetext::Processor(tagger, verbalizer)) {}
ItnProcessor::~ItnProcessor() = default;

std::string ItnProcessor::Normalize(const std::string& text) const {
  const std::string normalized = processor_->Normalize(LowerAscii(text));
  return RestoreUnchangedWordCase(text, normalized);
}

StatusOr<std::shared_ptr<const ItnProcessor>> CreateItnProcessor(
    const std::string& tagger, const std::string& verbalizer) {
  try {
    auto value = std::make_shared<ItnProcessor>(tagger, verbalizer);
    (void)value->Normalize("one hundred");
    return std::shared_ptr<const ItnProcessor>(std::move(value));
  } catch (const std::exception& e) {
    return Status::InvalidArgument(std::string("failed to initialize English ITN: ") + e.what());
  }
}

std::unique_ptr<AsrStream> WrapWithItn(
    std::unique_ptr<AsrStream> stream,
    std::shared_ptr<const ItnProcessor> processor) {
  if (!processor) return stream;
  return std::unique_ptr<AsrStream>(new ItnAsrStream(std::move(stream), std::move(processor)));
}

}  // namespace asr_sdk::internal
