#include "flashlight_decoder/contact_spelling_compiler.h"

#include <algorithm>
#include <cctype>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>

#include "flashlight_decoder/flashlight_decoder_resource.h"

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

StatusOr<std::vector<std::vector<int>>> ComposeLexiconSpellings(
    const FlashlightDecoderResource& resource,
    const std::vector<std::string>& units, int max_segmentations,
    int max_tokens) {
  std::vector<std::vector<int>> paths(1);
  for (const std::string& unit : units) {
    const auto it = resource.WordSpellings().find(unit);
    if (it == resource.WordSpellings().end() || it->second.empty()) {
      return std::vector<std::vector<int>>();
    }
    std::vector<std::vector<int>> next;
    for (const std::vector<int>& prefix : paths) {
      for (const std::vector<int>& spelling : it->second) {
        std::vector<int> candidate = prefix;
        candidate.insert(candidate.end(), spelling.begin(), spelling.end());
        if (static_cast<int>(candidate.size()) > max_tokens) {
          return Status::InvalidArgument(
              "contact spelling exceeds max_tokens_per_spoken_form");
        }
        next.push_back(std::move(candidate));
        if (static_cast<int>(next.size()) > max_segmentations) {
          return Status::InvalidArgument(
              "contact spelling has too many lexicon pronunciation "
              "combinations");
        }
      }
    }
    paths = std::move(next);
  }
  std::sort(paths.begin(), paths.end());
  paths.erase(std::unique(paths.begin(), paths.end()), paths.end());
  return paths;
}

struct SegmentationPiece {
  std::string text;
  int id = -1;
};

StatusOr<std::vector<std::vector<int>>> SegmentWithTokenTable(
    const FlashlightDecoderResource& resource, const std::string& text,
    int max_segmentations, int max_tokens) {
  const std::vector<std::string> words = SplitWhitespace(text);
  if (words.empty()) {
    return Status::InvalidArgument("spoken form is empty");
  }
  const std::string separator = resource.AmTokens().Token(resource.SilenceId());
  std::string target = separator;
  for (size_t i = 0; i < words.size(); ++i) {
    if (i > 0) {
      target += separator;
    }
    target += words[i];
  }

  std::vector<SegmentationPiece> pieces;
  for (int id = 0; id < resource.AmTokens().ModelVocabSize(); ++id) {
    if (id == resource.BlankId()) {
      continue;
    }
    const std::string& token = resource.AmTokens().Token(id);
    if (token.empty() || token[0] == '#') {
      continue;
    }
    // Ordinary special symbols cannot spell user text.  Byte pieces remain
    // valid because they can represent UTF-8 input exactly.
    if (token.front() == '<' && !IsByteToken(token) &&
        id != resource.SilenceId()) {
      continue;
    }
    pieces.push_back(SegmentationPiece{token, id});
  }
  std::sort(pieces.begin(), pieces.end(), [](const auto& left, const auto& right) {
    if (left.text.size() != right.text.size()) {
      return left.text.size() > right.text.size();
    }
    if (left.id != right.id) {
      return left.id < right.id;
    }
    return left.text < right.text;
  });

  std::vector<std::vector<int>> candidates;
  std::vector<int> current;
  bool overflow = false;
  std::function<void(size_t)> visit = [&](size_t offset) {
    if (overflow) {
      return;
    }
    if (offset == target.size()) {
      candidates.push_back(current);
      if (static_cast<int>(candidates.size()) > max_segmentations) {
        overflow = true;
      }
      return;
    }
    if (static_cast<int>(current.size()) >= max_tokens) {
      return;
    }
    for (const SegmentationPiece& piece : pieces) {
      if (target.compare(offset, piece.text.size(), piece.text) != 0) {
        continue;
      }
      current.push_back(piece.id);
      visit(offset + piece.text.size());
      current.pop_back();
      if (overflow) {
        return;
      }
    }
  };
  visit(0);
  if (overflow) {
    return Status::InvalidArgument(
        "contact spelling has too many token-table segmentations");
  }
  if (candidates.empty()) {
    return Status::InvalidArgument(
        "spoken form cannot be segmented with the existing AM token table");
  }
  std::sort(candidates.begin(), candidates.end(), [](const auto& left,
                                                       const auto& right) {
    if (left.size() != right.size()) {
      return left.size() < right.size();
    }
    return left < right;
  });
  candidates.erase(std::unique(candidates.begin(), candidates.end()),
                   candidates.end());
  return candidates;
}

StatusOr<std::vector<std::vector<int>>> CompileOneForm(
    const FlashlightDecoderResource& resource, const ContactSpokenForm& form,
    const ContactListOptions& options) {
  if (!form.am_tokens.empty()) {
    std::vector<int> ids;
    Status status = ValidateTokenIds(resource, form.am_tokens, &ids);
    if (!status.ok()) {
      return status;
    }
    if (static_cast<int>(ids.size()) > options.max_tokens_per_spoken_form) {
      return Status::InvalidArgument(
          "explicit AM spelling exceeds max_tokens_per_spoken_form");
    }
    return std::vector<std::vector<int>>{std::move(ids)};
  }

  const std::string normalized = NormalizeWhitespace(form.text);
  const std::vector<std::string> units = SplitWhitespace(normalized);
  auto lexicon_or = ComposeLexiconSpellings(
      resource, units, options.max_token_segmentations_per_form,
      options.max_tokens_per_spoken_form);
  if (!lexicon_or.ok()) {
    return lexicon_or.status();
  }
  if (!lexicon_or.value().empty()) {
    return std::move(lexicon_or).value();
  }
  return SegmentWithTokenTable(resource, normalized,
                               options.max_token_segmentations_per_form,
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
  for (const ContactEntry& entry : config.contacts) {
    const std::string id = Trim(entry.contact_id);
    const std::string display_name = NormalizeWhitespace(entry.display_name);
    if (id.empty()) {
      return Status::InvalidArgument("contact_id must not be empty");
    }
    if (display_name.empty()) {
      return Status::InvalidArgument("display_name must not be empty for " +
                                     id);
    }
    auto [it, inserted] = contacts.emplace(id, InputContact{display_name, {}});
    if (!inserted && it->second.display_name != display_name) {
      return Status::InvalidArgument(
          "repeated contact_id has conflicting display_name: " + id);
    }
    it->second.forms.insert(it->second.forms.end(), entry.spoken_forms.begin(),
                            entry.spoken_forms.end());
  }
  if (static_cast<int>(contacts.size()) > options.max_contacts) {
    return Status::InvalidArgument("contact count exceeds max_contacts");
  }

  std::vector<CompiledContactSpelling> compiled;
  for (const auto& [contact_id, contact] : contacts) {
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
      auto token_sequences_or = CompileOneForm(resource, form, options);
      if (!token_sequences_or.ok()) {
        const std::string message = "contact " + contact_id + " form '" +
                                    spoken_form + "': " +
                                    token_sequences_or.status().message();
        if (!options.skip_unencodable_forms) {
          return Status::InvalidArgument(message);
        }
        if (diagnostics != nullptr) {
          diagnostics->push_back("skipped " + message);
        }
        continue;
      }
      for (std::vector<int>& token_ids : token_sequences_or.value()) {
        CompiledContactSpelling spelling;
        spelling.contact_id = contact_id;
        spelling.display_name = contact.display_name;
        spelling.spoken_form = spoken_form;
        spelling.token_ids = std::move(token_ids);
        spelling.logical_word_count = LogicalWordCount(form, spoken_form);
        compiled.push_back(std::move(spelling));
        if (static_cast<int>(compiled.size()) > options.max_total_dynamic_forms) {
          return Status::InvalidArgument(
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
    const std::string key = spelling.contact_id + "\x1e" +
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
