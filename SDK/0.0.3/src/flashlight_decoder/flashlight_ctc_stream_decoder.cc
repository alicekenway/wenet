#include "flashlight_decoder/flashlight_ctc_stream_decoder.h"

#include <algorithm>
#include <exception>
#include <limits>
#include <memory>
#include <sstream>
#include <unordered_map>

#include "flashlight/lib/text/decoder/LexiconDecoder.h"
#include "flashlight_decoder/flashlight_result_mapper.h"

namespace asr_sdk::internal::flashlight_decoder {
namespace {

fl::lib::text::LexiconDecoderOptions MakeDecoderOptions(
    const FlashlightDecoderOptions& options) {
  fl::lib::text::LexiconDecoderOptions out;
  out.beamSize = options.beam_size;
  out.beamSizeToken = options.beam_size_token;
  out.beamThreshold = options.beam_threshold;
  out.lmWeight = options.lm_weight;
  out.wordScore = options.word_score;
  out.unkScore = options.allow_unk
                     ? options.unk_score
                     : -std::numeric_limits<float>::infinity();
  out.silScore = options.sil_score;
  out.logAdd = options.log_add;
  out.criterionType = fl::lib::text::CriterionType::CTC;
  return out;
}

std::string KeyForWords(const std::vector<DecodedWord>& words) {
  std::string key;
  for (const DecodedWord& word : words) {
    key += std::to_string(word.word_id);
    key.push_back('\x1f');
  }
  return key;
}

std::vector<DecodedHypothesis> DeduplicateMapped(
    std::vector<DecodedHypothesis> hyps, int limit) {
  std::vector<DecodedHypothesis> out;
  std::unordered_map<std::string, size_t> best_by_key;
  for (DecodedHypothesis& hyp : hyps) {
    const std::string key = KeyForWords(hyp.mapped_words);
    const auto it = best_by_key.find(key);
    if (it == best_by_key.end()) {
      best_by_key[key] = out.size();
      out.push_back(std::move(hyp));
      continue;
    }
    DecodedHypothesis& existing = out[it->second];
    if (hyp.total_score > existing.total_score) {
      existing = std::move(hyp);
    }
  }
  std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
    return a.total_score > b.total_score;
  });
  if (limit > 0 && static_cast<int>(out.size()) > limit) {
    out.resize(static_cast<size_t>(limit));
  }
  return out;
}

Status ExceptionStatus(const char* where, const std::exception& e) {
  return Status::Internal(std::string(where) + ": " + e.what());
}

}  // namespace

struct FlashlightCtcStreamDecoder::Impl {
  explicit Impl(FlashlightDecoderResourcePtr resource_in)
      : resource(std::move(resource_in)) {}

  FlashlightDecoderResourcePtr resource;
  std::unique_ptr<fl::lib::text::LexiconDecoder> decoder;
  bool started = false;
  bool finalized = false;
};

FlashlightCtcStreamDecoder::FlashlightCtcStreamDecoder(
    FlashlightDecoderResourcePtr resource)
    : impl_(std::make_unique<Impl>(std::move(resource))) {}

FlashlightCtcStreamDecoder::~FlashlightCtcStreamDecoder() = default;

Status FlashlightCtcStreamDecoder::Start() {
  if (!impl_->resource) {
    return Status::FailedPrecondition("Flashlight decoder resource is null");
  }
  if (impl_->started && !impl_->finalized) {
    return Status::Ok();
  }
  try {
    impl_->decoder = std::make_unique<fl::lib::text::LexiconDecoder>(
        MakeDecoderOptions(impl_->resource->Options()),
        impl_->resource->LexiconTrie(), impl_->resource->WordLm(),
        impl_->resource->SilenceId(), impl_->resource->BlankId(),
        impl_->resource->UnknownWordId(), std::vector<float>{}, false);
    impl_->decoder->decodeBegin();
    impl_->started = true;
    impl_->finalized = false;
    return Status::Ok();
  } catch (const std::exception& e) {
    return ExceptionStatus("Flashlight decodeBegin failed", e);
  }
}

Status FlashlightCtcStreamDecoder::DecodeChunk(const float* data, int frames,
                                               int vocab_size) {
  if (data == nullptr) {
    return Status::InvalidArgument("DecodeChunk data is null");
  }
  if (frames < 0 || vocab_size <= 0) {
    return Status::InvalidArgument("DecodeChunk received invalid shape");
  }
  if (impl_->finalized) {
    return Status::FailedPrecondition("DecodeChunk after Finalize");
  }
  Status status = Start();
  if (!status.ok()) {
    return status;
  }
  if (vocab_size != impl_->resource->AmTokens().ModelVocabSize()) {
    return Status::InvalidArgument(
        "DecodeChunk vocab size does not match tokens.txt/model vocab");
  }
  try {
    if (frames > 0) {
      impl_->decoder->decodeStep(data, frames, vocab_size);
    }
    return Status::Ok();
  } catch (const std::exception& e) {
    return ExceptionStatus("Flashlight decodeStep failed", e);
  }
}

StatusOr<DecodedHypothesis> FlashlightCtcStreamDecoder::PartialResult() const {
  if (!impl_->started || !impl_->decoder) {
    return Status::FailedPrecondition("PartialResult before Start");
  }
  try {
    return ConvertFlashlightResult(impl_->decoder->getBestHypothesis(),
                                   *impl_->resource);
  } catch (const std::exception& e) {
    return ExceptionStatus("Flashlight partial result failed", e);
  }
}

StatusOr<std::vector<DecodedHypothesis>>
FlashlightCtcStreamDecoder::Finalize() {
  if (impl_->finalized) {
    return Status::FailedPrecondition("Finalize called twice");
  }
  Status status = Start();
  if (!status.ok()) {
    return status;
  }
  try {
    impl_->decoder->decodeEnd();
    impl_->finalized = true;
    std::vector<DecodedHypothesis> hyps;
    const auto results = impl_->decoder->getAllFinalHypothesis();
    hyps.reserve(results.size());
    for (const auto& result : results) {
      hyps.push_back(ConvertFlashlightResult(result, *impl_->resource));
    }
    return DeduplicateMapped(std::move(hyps), impl_->resource->Options().nbest);
  } catch (const std::exception& e) {
    return ExceptionStatus("Flashlight final result failed", e);
  }
}

Status FlashlightCtcStreamDecoder::Reset() {
  impl_->decoder.reset();
  impl_->started = false;
  impl_->finalized = false;
  return Status::Ok();
}

}  // namespace asr_sdk::internal::flashlight_decoder
