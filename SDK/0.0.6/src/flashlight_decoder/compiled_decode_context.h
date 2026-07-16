#ifndef ASR_SDK_SRC_FLASHLIGHT_DECODER_COMPILED_DECODE_CONTEXT_H_
#define ASR_SDK_SRC_FLASHLIGHT_DECODER_COMPILED_DECODE_CONTEXT_H_

#include <memory>
#include <string>
#include <vector>

#include "asr_sdk/decode_context.h"
#include "asr_sdk/status.h"
#include "flashlight_decoder/dynamic_contact_lexicon.h"
#include "flashlight_decoder/flashlight_decoder_resource.h"

namespace asr_sdk::internal::flashlight_decoder {

// A per-engine identity token prevents accidental cross-engine context reuse.
struct ContextOwnerToken {};

class CompiledDecodeContext final : public DecodeContext {
 public:
  static StatusOr<std::shared_ptr<const CompiledDecodeContext>> Compile(
      FlashlightDecoderResourcePtr base_resource,
      const DecodeContextConfig& config,
      std::shared_ptr<const ContextOwnerToken> owner);

  static std::shared_ptr<const CompiledDecodeContext> MakeEmpty(
      std::shared_ptr<const ContextOwnerToken> owner);

  bool IsEmpty() const { return empty_; }
  bool BelongsTo(const std::shared_ptr<const ContextOwnerToken>& owner) const {
    return owner_ == owner;
  }
  // The immutable package/main-LM resource.  It remains separate from the
  // contact overlay so one stream can run both decoder searches from the same
  // AM emissions.
  const FlashlightDecoderResourcePtr& DecoderResource() const {
    return decoder_resource_;
  }
  // Non-null only when at least one runtime contact form compiled and the
  // package supplies a contact-domain LM.
  const FlashlightDecoderResourcePtr& ContactDecoderResource() const {
    return contact_decoder_resource_;
  }
  const std::shared_ptr<const DynamicContactLexicon>& DynamicContacts() const {
    return dynamic_contacts_;
  }
  const std::string& Fingerprint() const { return fingerprint_; }
  const std::vector<std::string>& Diagnostics() const { return diagnostics_; }

 private:
  explicit CompiledDecodeContext(std::shared_ptr<const ContextOwnerToken> owner)
      : owner_(std::move(owner)) {}

  std::shared_ptr<const ContextOwnerToken> owner_;
  // Base/main decoder resource.  Kept in this context for the dual-decoder
  // stream, even though it is normally shared by the engine as well.
  FlashlightDecoderResourcePtr decoder_resource_;
  FlashlightDecoderResourcePtr contact_decoder_resource_;
  std::shared_ptr<const DynamicContactLexicon> dynamic_contacts_;
  std::string fingerprint_;
  std::vector<std::string> diagnostics_;
  bool empty_ = true;
};

}  // namespace asr_sdk::internal::flashlight_decoder

#endif  // ASR_SDK_SRC_FLASHLIGHT_DECODER_COMPILED_DECODE_CONTEXT_H_
