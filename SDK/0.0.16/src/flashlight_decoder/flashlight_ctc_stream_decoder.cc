#include "flashlight_decoder/flashlight_ctc_stream_decoder.h"

#include <exception>
#include <limits>
#include <memory>

#include "flashlight_decoder/flashlight_result_mapper.h"
#include "flashlight_decoder/multi_trie_lexicon_decoder.h"

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

Status ExceptionStatus(const char* where, const std::exception& e) {
  return Status::Internal(std::string(where) + ": " + e.what());
}

}  // namespace

struct FlashlightCtcStreamDecoder::Impl {
  explicit Impl(FlashlightDecoderResourcePtr resource_in)
      : resource(std::move(resource_in)) {}

  FlashlightDecoderResourcePtr resource;
  std::unique_ptr<fl::lib::text::Decoder> decoder;
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
    impl_->decoder = std::make_unique<MultiTrieLexiconDecoder>(
        MakeDecoderOptions(impl_->resource->Options()),
        impl_->resource->Lexicon(), impl_->resource->OverlayTrie(),
        impl_->resource->WordLm(), impl_->resource->SilenceId(),
        impl_->resource->BlankId(), impl_->resource->UnknownWordId(),
        std::vector<float>{}, false, impl_->resource->HasSlotContext());
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
    // Preserve Flashlight's historical streaming semantics: this returns the
    // best path ending at a completed lexicon word rather than an arbitrary
    // state from the current beam.
    return ConvertFlashlightResult(impl_->decoder->getBestHypothesis(),
                                   *impl_->resource);
  } catch (const std::exception& e) {
    return ExceptionStatus("Flashlight partial result failed", e);
  }
}

StatusOr<std::vector<DecodedHypothesis>>
FlashlightCtcStreamDecoder::PartialResults(int limit) const {
  if (!impl_->started || !impl_->decoder) {
    return Status::FailedPrecondition("PartialResults before Start");
  }
  try {
    std::vector<DecodedHypothesis> hyps;
    // Flashlight's getAllFinalHypothesis() returns the current frame's beam;
    // decodeEnd() is only needed to finalize the final-frame bookkeeping.
    const auto results = impl_->decoder->getAllFinalHypothesis();
    hyps.reserve(results.size());
    for (const auto& result : results) {
      hyps.push_back(ConvertFlashlightResult(result, *impl_->resource));
    }
    return DeduplicateDecodedHypotheses(
        std::move(hyps), limit,
        impl_->resource->HasSlotContext());
  } catch (const std::exception& e) {
    return ExceptionStatus("Flashlight partial results failed", e);
  }
}

StatusOr<std::vector<DecodedHypothesis>>
FlashlightCtcStreamDecoder::Finalize(int limit) {
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
    return DeduplicateDecodedHypotheses(
        std::move(hyps), limit,
        impl_->resource->HasSlotContext());
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
