#ifndef ASR_SDK_DECODE_CONTEXT_H_
#define ASR_SDK_DECODE_CONTEXT_H_

#include <string>
#include <vector>

#include "asr_sdk/export.h"

namespace asr_sdk {

// A visible form that may be used to recognize a contact.  `am_tokens`, when
// supplied, is an exact spelling in the package's existing CTC token table.
struct ContactSpokenForm {
  std::string text;
  std::vector<std::string> am_tokens;
  int logical_word_count = 0;
};

// Contact IDs are opaque application values.  They are copied into an
// immutable DecodeContext and are never written into a model package.
struct ContactEntry {
  std::string contact_id;
  std::string display_name;
  std::vector<ContactSpokenForm> spoken_forms;
};

// Limits applied while compiling a runner-provided contact list. Contact LM
// scoring is configured by the immutable model package, not per context.
struct ContactListOptions {
  int max_contacts = 10000;
  int max_spoken_forms_per_contact = 8;
  int max_tokens_per_spoken_form = 64;
  int max_total_dynamic_forms = 50000;
  int max_token_segmentations_per_form = 16;

  // If false, an unencodable form fails context compilation.  If true, only
  // that form is skipped and the compiled context records a diagnostic.
  bool skip_unencodable_forms = false;
};

struct DecodeContextConfig {
  std::vector<ContactEntry> contacts;
  ContactListOptions contact_list;
};

// Contexts are created only through AsrEngine::CompileDecodeContext().
// Implementations are immutable and may be safely shared by multiple streams.
class ASR_SDK_API DecodeContext {
 public:
  virtual ~DecodeContext() = default;
};

}  // namespace asr_sdk

#endif  // ASR_SDK_DECODE_CONTEXT_H_
