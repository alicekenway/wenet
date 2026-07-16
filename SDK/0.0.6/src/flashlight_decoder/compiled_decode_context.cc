#include "flashlight_decoder/compiled_decode_context.h"

#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <sstream>
#include <tuple>
#include <utility>

#include "flashlight_decoder/contextual_contact_lm.h"
#include "flashlight_decoder/contact_spelling_compiler.h"

namespace asr_sdk::internal::flashlight_decoder {
namespace {

void HashBytes(uint64_t* hash, const std::string& value) {
  constexpr uint64_t kFnvPrime = 1099511628211ULL;
  for (unsigned char byte : value) {
    *hash ^= byte;
    *hash *= kFnvPrime;
  }
  *hash ^= 0xff;
  *hash *= kFnvPrime;
}

std::string MakeFingerprint(const FlashlightDecoderResource& base,
                            const DynamicContactLexicon& contacts) {
  uint64_t hash = 1469598103934665603ULL;
  HashBytes(&hash, std::to_string(base.BaseWordCount()));
  HashBytes(&hash, base.ContactClassWord());
  HashBytes(&hash, std::to_string(base.ContactLmWeight()));
  HashBytes(&hash, std::to_string(base.ContactLmAccumulationFactor()));
  for (const RuntimeContactForm& form : contacts.Forms()) {
    HashBytes(&hash, form.spoken_form);
    HashBytes(&hash, form.visible_text);
    HashBytes(&hash, std::to_string(form.logical_word_count));
    for (int token_id : form.token_ids) {
      HashBytes(&hash, std::to_string(token_id));
    }
    for (const RuntimeContactCandidate& candidate : form.candidates) {
      HashBytes(&hash, candidate.contact_id);
      HashBytes(&hash, candidate.display_name);
    }
  }
  std::ostringstream out;
  out << std::hex << std::setfill('0') << std::setw(16) << hash;
  return out.str();
}

}  // namespace

std::shared_ptr<const CompiledDecodeContext> CompiledDecodeContext::MakeEmpty(
    std::shared_ptr<const ContextOwnerToken> owner) {
  return std::shared_ptr<const CompiledDecodeContext>(
      new CompiledDecodeContext(std::move(owner)));
}

StatusOr<std::shared_ptr<const CompiledDecodeContext>>
CompiledDecodeContext::Compile(FlashlightDecoderResourcePtr base_resource,
                               const DecodeContextConfig& config,
                               std::shared_ptr<const ContextOwnerToken> owner) {
  if (!base_resource || !owner) {
    return Status::Internal("missing Flashlight context compilation resources");
  }
  if (config.contacts.empty()) {
    return MakeEmpty(std::move(owner));
  }
  if (!base_resource->SupportsRuntimeContacts()) {
    return Status::FailedPrecondition(
        "runtime contacts require a package with contact_class_word and "
        "contact_lm");
  }
  const double contact_lm_weight = base_resource->ContactLmWeight();
  if (!std::isfinite(contact_lm_weight) || contact_lm_weight <= 0.0) {
    return Status::FailedPrecondition(
        "runtime contacts require finite contact_lm_weight > 0");
  }

  try {
    auto out = std::shared_ptr<CompiledDecodeContext>(
        new CompiledDecodeContext(std::move(owner)));
    out->decoder_resource_ = base_resource;
    auto spellings_or =
        ContactSpellingCompiler::Compile(*base_resource, config,
                                         &out->diagnostics_);
    if (!spellings_or.ok()) {
      return spellings_or.status();
    }
    auto contacts_or = DynamicContactLexicon::Create(
        *base_resource, std::move(spellings_or).value());
    if (!contacts_or.ok()) {
      return contacts_or.status();
    }
    out->dynamic_contacts_ = std::move(contacts_or).value();
    if (out->dynamic_contacts_->Forms().empty()) {
      out->diagnostics_.push_back(
          "no encodable runtime contact forms; using the base decoder");
      return std::shared_ptr<const CompiledDecodeContext>(std::move(out));
    }

    auto contextual_lm = std::make_shared<ContextualContactLm>(
        base_resource->WordLm(), base_resource->ContactWordLm(),
        out->dynamic_contacts_,
        base_resource->ContactClassWordId(),
        base_resource->Options().lm_weight, contact_lm_weight,
        base_resource->ContactLmAccumulationFactor(),
        static_cast<float>(base_resource->Options().word_score));
    auto combined_trie =
        out->dynamic_contacts_->BuildCombinedTrie(*base_resource);
    out->contact_decoder_resource_ =
        FlashlightDecoderResource::CreateContactContextResource(
            *base_resource, std::move(combined_trie), std::move(contextual_lm),
            out->dynamic_contacts_);
    out->fingerprint_ = MakeFingerprint(*base_resource, *out->dynamic_contacts_);
    out->empty_ = false;
    return std::shared_ptr<const CompiledDecodeContext>(std::move(out));
  } catch (const std::exception& e) {
    return Status::Internal(std::string("failed to compile runtime contacts: ") +
                            e.what());
  }
}

}  // namespace asr_sdk::internal::flashlight_decoder
