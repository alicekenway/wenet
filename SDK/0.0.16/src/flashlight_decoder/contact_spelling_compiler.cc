#include "flashlight_decoder/contact_spelling_compiler.h"

#include <algorithm>
#include <cctype>
#include <exception>
#include <unordered_map>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "flashlight_decoder/flashlight_decoder_resource.h"
#include "flashlight_decoder/word_spelling_generator.h"

namespace asr_sdk::internal::flashlight_decoder {
namespace {

std::string Trim(std::string value) {
  size_t first = 0;
  while (first < value.size() &&
         std::isspace(static_cast<unsigned char>(value[first]))) {
    ++first;
  }
  size_t last = value.size();
  while (last > first &&
         std::isspace(static_cast<unsigned char>(value[last - 1]))) {
    --last;
  }
  return value.substr(first, last - first);
}

std::string NormalizeWhitespace(const std::string& value) {
  std::istringstream input(value);
  std::string output;
  std::string item;
  while (input >> item) {
    if (!output.empty()) {
      output.push_back(' ');
    }
    output += item;
  }
  return output;
}

std::vector<std::string> SplitWhitespace(const std::string& value) {
  std::istringstream input(value);
  std::vector<std::string> items;
  std::string item;
  while (input >> item) {
    items.push_back(item);
  }
  return items;
}

std::string IdKey(const std::vector<int>& ids) {
  std::string key;
  for (int id : ids) {
    key += std::to_string(id);
    key.push_back('\x1f');
  }
  return key;
}

bool IsByteToken(const std::string& token) {
  return token.size() == 6 && token.rfind("<0x", 0) == 0 &&
         token.back() == '>';
}

Status ValidateTokenIds(const FlashlightDecoderResource& resource,
                        const std::vector<std::string>& tokens,
                        std::vector<int>* ids) {
  if (tokens.empty()) {
    return Status::InvalidArgument("explicit AM token sequence is empty");
  }
  ids->clear();
  ids->reserve(tokens.size());
  for (const std::string& token : tokens) {
    if (token.empty()) {
      return Status::InvalidArgument("explicit AM token is empty");
    }
    if (!resource.AmTokens().Contains(token)) {
      return Status::InvalidArgument("unknown explicit AM token: " + token);
    }
    const int id = resource.AmTokens().Id(token);
    if (id == resource.BlankId()) {
      return Status::InvalidArgument(
          "blank token is not allowed in a contact spelling: " + token);
    }
    if (id < 0 || id >= resource.AmTokens().ModelVocabSize()) {
      return Status::InvalidArgument(
          "explicit AM token is outside the ONNX vocabulary: " + token);
    }
    if (!token.empty() && token[0] == '#') {
      return Status::InvalidArgument(
          "non-emitting AM token is not allowed in a contact spelling: " +
          token);
    }
    const std::string& separator =
        resource.AmTokens().Token(resource.SilenceId());
    if (token.front() == '<' && !IsByteToken(token) && token != separator) {
      return Status::InvalidArgument(
          "special AM token is not allowed in a contact spelling: " + token);
    }
    ids->push_back(id);
  }
  return Status::Ok();
}

StatusOr<std::vector<std::vector<int>>> CompileOneForm(
    const FlashlightDecoderResource& resource, const ContactSpokenForm& form,
    const ContactListOptions& options, const WordSpellingGenerator* generator,
    std::unordered_map<std::string, GeneratedWordSpellings>* cache,
    size_t remaining_paths, std::vector<std::string>* diagnostics) {
  if (!form.am_tokens.empty()) {
    std::vector<int> ids;
    Status status = ValidateTokenIds(resource, form.am_tokens, &ids);
    if (!status.ok()) {
      return status;
    }
    if (static_cast<int>(ids.size()) > options.max_tokens_per_spoken_form) {
      return Status::FailedPrecondition(
          "explicit AM spelling exceeds max_tokens_per_spoken_form");
    }
    return std::vector<std::vector<int>>{std::move(ids)};
  }

  const std::string normalized = NormalizeWhitespace(form.text);
  const std::vector<std::string> units = SplitWhitespace(normalized);
  std::vector<const GeneratedWordSpellings*> word_paths;
  for (const auto& word : units) {
    auto found = cache->find(word);
    if (found == cache->end()) {
      try {
        found = cache->emplace(word, generator->Generate(word)).first;
      } catch (const std::invalid_argument& e) {
        return Status::InvalidArgument(e.what());
      }
      if (diagnostics) {
        diagnostics->insert(diagnostics->end(),
            found->second.diagnostics.begin(), found->second.diagnostics.end());
        diagnostics->push_back("word " + word + " paths=" +
                               std::to_string(found->second.paths.size()));
      }
    }
    if (found->second.paths.empty()) {
      return Status::InvalidArgument("contact word has no valid token path: " + word);
    }
    word_paths.push_back(&found->second);
  }
  return CombineWordSpellings(word_paths,
      std::min(remaining_paths, static_cast<size_t>(options.max_token_segmentations_per_form)),
      options.max_tokens_per_spoken_form);
}

int LogicalWordCount(const ContactSpokenForm& form,
                     const std::string& normalized) {
  if (form.logical_word_count > 0) {
    return form.logical_word_count;
  }
  const int inferred = static_cast<int>(SplitWhitespace(normalized).size());
  return inferred > 0 ? inferred : 1;
}

struct InputContact {
  std::string slot_token;
  int slot_word_id = -1;
  std::string display_name;
  std::vector<ContactSpokenForm> forms;
};

}  // namespace

StatusOr<std::vector<CompiledContactSpelling>> ContactSpellingCompiler::Compile(
    const FlashlightDecoderResource& resource, const DecodeContextConfig& config,
    std::vector<std::string>* diagnostics) {
  const ContactListOptions& options = config.contact_list;
  if (options.max_contacts <= 0 || options.max_spoken_forms_per_contact <= 0 ||
      options.max_tokens_per_spoken_form <= 0 ||
      options.max_total_dynamic_forms <= 0 ||
      options.max_token_segmentations_per_form <= 0) {
    return Status::InvalidArgument("runtime contact limits must be positive");
  }

  std::map<std::string, InputContact> contacts;
  auto add_value = [&](const std::string& slot_token,
                       const std::string& raw_id,
                       const std::string& raw_display,
                       const std::vector<ContactSpokenForm>& forms) -> Status {
    const std::string id = Trim(raw_id);
    const std::string display_name = NormalizeWhitespace(raw_display);
    if (slot_token.empty() || !resource.OutputWords().Contains(slot_token)) {
      return Status::InvalidArgument("unknown runtime slot token: " + slot_token);
    }
    if (!resource.SupportsRuntimeSlot(slot_token)) {
      return Status::FailedPrecondition(
          "runtime slot has no configured bias LM: " + slot_token);
    }
    if (id.empty()) return Status::InvalidArgument("value_id must not be empty");
    if (display_name.empty()) {
      return Status::InvalidArgument("display_name must not be empty for " + id);
    }
    const std::string key = slot_token + "\x1f" + id;
    auto [it, inserted] = contacts.emplace(
        key, InputContact{slot_token, resource.OutputWords().Id(slot_token),
                          display_name, {}});
    if (!inserted && it->second.display_name != display_name) {
      return Status::InvalidArgument(
          "repeated slot value has conflicting display_name: " + id);
    }
    it->second.forms.insert(it->second.forms.end(), forms.begin(), forms.end());
    return Status::Ok();
  };
  for (const ContactEntry& entry : config.contacts) {
    Status status = add_value("<CONTACT>", entry.contact_id,
                              entry.display_name, entry.spoken_forms);
    if (!status.ok()) return status;
  }
  for (const SlotClass& slot : config.slots) {
    for (const SlotValueEntry& value : slot.values) {
      Status status = add_value(slot.slot_token, value.value_id,
                                value.display_name, value.spoken_forms);
      if (!status.ok()) return status;
    }
  }
  if (static_cast<int>(contacts.size()) > options.max_contacts) {
    return Status::InvalidArgument("contact count exceeds max_contacts");
  }

  // Initialize once, and only when automatic text forms are actually present.
  // Package/model errors must not turn into a silently skipped contact list.
  const WordSpellingGenerator* generator = nullptr;
  for (const auto& item : contacts) {
    for (const auto& form : item.second.forms) {
      if (form.am_tokens.empty() && !generator) {
        try {
          generator = &resource.ContactWordSpeller();
        } catch (const std::exception& e) {
          return Status::FailedPrecondition(e.what());
        }
        if (diagnostics) diagnostics->insert(diagnostics->end(),
            generator->Diagnostics().begin(), generator->Diagnostics().end());
      }
    }
  }
  std::unordered_map<std::string, GeneratedWordSpellings> word_cache;
  std::vector<CompiledContactSpelling> compiled;
  for (const auto& [key, contact] : contacts) {
    const std::string contact_id = key.substr(key.find('\x1f') + 1);
    if (contact.forms.empty()) {
      return Status::InvalidArgument("contact has no spoken forms: " +
                                     contact_id);
    }
    if (static_cast<int>(contact.forms.size()) >
        options.max_spoken_forms_per_contact) {
      return Status::InvalidArgument(
          "contact has more than max_spoken_forms_per_contact forms: " +
          contact_id);
    }

    const size_t count_before_contact = compiled.size();
    for (const ContactSpokenForm& form : contact.forms) {
      const std::string spoken_form = NormalizeWhitespace(form.text);
      if (spoken_form.empty()) {
        return Status::InvalidArgument("spoken form must not be empty for " +
                                       contact_id);
      }
      if (form.logical_word_count < 0) {
        return Status::InvalidArgument(
            "logical_word_count must be zero or positive for " + contact_id);
      }
      auto token_sequences_or = CompileOneForm(resource, form, options,
          generator, &word_cache, options.max_total_dynamic_forms - compiled.size(),
          diagnostics);
      if (!token_sequences_or.ok()) {
        const std::string message = "contact " + contact_id + " form '" +
                                    spoken_form + "': " +
                                    token_sequences_or.status().message();
        if (!options.skip_unencodable_forms ||
            token_sequences_or.status().code() != StatusCode::kInvalidArgument) {
          return Status(token_sequences_or.status().code(), message);
        }
        if (diagnostics != nullptr) {
          diagnostics->push_back("skipped " + message);
        }
        continue;
      }
      if (diagnostics) diagnostics->push_back("contact " + contact_id +
          " form '" + spoken_form + "' paths=" +
          std::to_string(token_sequences_or.value().size()));
      for (std::vector<int>& token_ids : token_sequences_or.value()) {
        CompiledContactSpelling spelling;
        spelling.slot_token = contact.slot_token;
        spelling.slot_word_id = contact.slot_word_id;
        spelling.contact_id = contact_id;
        spelling.display_name = contact.display_name;
        spelling.spoken_form = spoken_form;
        spelling.token_ids = std::move(token_ids);
        spelling.logical_word_count = LogicalWordCount(form, spoken_form);
        compiled.push_back(std::move(spelling));
        if (static_cast<int>(compiled.size()) > options.max_total_dynamic_forms) {
          return Status::FailedPrecondition(
              "compiled contact forms exceed max_total_dynamic_forms");
        }
      }
    }
    if (compiled.size() == count_before_contact &&
        options.skip_unencodable_forms && diagnostics != nullptr) {
      diagnostics->push_back("contact " + contact_id +
                             " has no encodable spoken forms");
    }
  }

  std::sort(compiled.begin(), compiled.end(), [](const auto& left,
                                                   const auto& right) {
    if (left.token_ids != right.token_ids) {
      return left.token_ids < right.token_ids;
    }
    if (left.slot_token != right.slot_token) {
      return left.slot_token < right.slot_token;
    }
    if (left.contact_id != right.contact_id) {
      return left.contact_id < right.contact_id;
    }
    if (left.spoken_form != right.spoken_form) {
      return left.spoken_form < right.spoken_form;
    }
    return left.display_name < right.display_name;
  });
  std::unordered_set<std::string> seen;
  std::vector<CompiledContactSpelling> deduplicated;
  deduplicated.reserve(compiled.size());
  for (CompiledContactSpelling& spelling : compiled) {
    const std::string key = spelling.slot_token + "\x1e" +
                            spelling.contact_id + "\x1e" +
                            spelling.display_name + "\x1e" +
                            spelling.spoken_form + "\x1e" +
                            IdKey(spelling.token_ids) + "\x1e" +
                            std::to_string(spelling.logical_word_count);
    if (seen.insert(key).second) {
      deduplicated.push_back(std::move(spelling));
    }
  }
  return deduplicated;
}

}  // namespace asr_sdk::internal::flashlight_decoder
