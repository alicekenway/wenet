#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "asr_sdk/config.h"
#include "flashlight_decoder/compact_lexicon.h"
#include "flashlight_decoder/compact_lexicon_builder.h"
#include "flashlight_decoder/flashlight_decoder_resource.h"
#include "package/model_package.h"

namespace {

namespace fs = std::filesystem;
using asr_sdk::internal::ModelPackage;
using asr_sdk::internal::flashlight_decoder::CompactLexicon;
using asr_sdk::internal::flashlight_decoder::CompactLexiconBuildStats;
using asr_sdk::internal::flashlight_decoder::FlashlightDecoderResource;

struct Arguments {
  std::string command;
  fs::path source_package;
  fs::path input;
  fs::path output;
  fs::path report;
};

void Usage(const char* program) {
  std::cerr
      << "usage:\n"
      << "  " << program
      << " compile --source-package DIR --output lexicon.bin"
         " [--report report.json]\n"
      << "  " << program
      << " verify --source-package DIR --input lexicon.bin\n"
      << "  " << program << " inspect --input lexicon.bin\n";
}

Arguments ParseArguments(int argc, char** argv) {
  if (argc < 2) throw std::invalid_argument("missing command");
  Arguments arguments;
  arguments.command = argv[1];
  for (int index = 2; index < argc; ++index) {
    const std::string option = argv[index];
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + option);
    }
    const fs::path value = argv[++index];
    if (option == "--source-package") {
      arguments.source_package = value;
    } else if (option == "--input") {
      arguments.input = value;
    } else if (option == "--output") {
      arguments.output = value;
    } else if (option == "--report") {
      arguments.report = value;
    } else {
      throw std::invalid_argument("unknown option: " + option);
    }
  }
  if (arguments.command == "compile") {
    if (arguments.source_package.empty() || arguments.output.empty() ||
        !arguments.input.empty()) {
      throw std::invalid_argument("compile requires source package and output");
    }
  } else if (arguments.command == "verify") {
    if (arguments.source_package.empty() || arguments.input.empty() ||
        !arguments.output.empty() || !arguments.report.empty()) {
      throw std::invalid_argument("verify requires source package and input");
    }
  } else if (arguments.command == "inspect") {
    if (arguments.input.empty() || !arguments.source_package.empty() ||
        !arguments.output.empty() || !arguments.report.empty()) {
      throw std::invalid_argument("inspect requires only input");
    }
  } else {
    throw std::invalid_argument("unknown command: " + arguments.command);
  }
  return arguments;
}

ModelPackage LoadSourcePackage(const fs::path& directory) {
  asr_sdk::EngineConfig config;
  config.model_dir = directory.string();
  auto package_or = asr_sdk::internal::LoadModelPackage(config);
  if (!package_or.ok()) {
    throw std::runtime_error(package_or.status().ToString());
  }
  ModelPackage package = std::move(package_or).value();
  if (package.decoder_type != "flashlight_lexicon_kenlm") {
    throw std::invalid_argument(
        "source package must use decoder_type=flashlight_lexicon_kenlm");
  }
  for (const auto& [path, label] :
       {std::pair{package.tokens_txt, "tokens.txt"},
        std::pair{package.words_txt, "words.txt"},
        std::pair{package.lexicon_txt, "lexicon.txt"},
        std::pair{package.lm_search_json, "lm_search.json"}}) {
    if (path.empty() || !fs::is_regular_file(path)) {
      throw std::runtime_error(std::string(label) + " not found: " +
                               path.string());
    }
  }
  return package;
}

std::shared_ptr<FlashlightDecoderResource> BuildLegacyResource(
    const ModelPackage& package) {
  return std::make_shared<FlashlightDecoderResource>(
      package.tokens_txt, package.words_txt, package.lexicon_txt,
      package.fixed_lms, package.output_mapping_txt,
      package.final_output_mapping_txt, package.flashlight_options,
      package.blank_token, package.sil_token, package.unk_word,
      package.length_penalty, false, 0);
}

void WriteReport(const fs::path& path, const fs::path& source_package,
                 const fs::path& output,
                 const CompactLexiconBuildStats& stats) {
  if (path.empty()) return;
  if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
  std::ofstream report(path, std::ios::trunc);
  if (!report) {
    throw std::runtime_error("failed to open report: " + path.string());
  }
  report << "{\n"
         << "  \"format\": \"compact_trie_v1\",\n"
         << "  \"compiler_sdk_version\": \"0.0.16\",\n"
         << "  \"source_package\": \"" << source_package.string()
         << "\",\n"
         << "  \"output\": \"" << output.string() << "\",\n"
         << "  \"token_count\": " << stats.token_count << ",\n"
         << "  \"word_count\": " << stats.word_count << ",\n"
         << "  \"node_count\": " << stats.node_count << ",\n"
         << "  \"edge_count\": " << stats.edge_count << ",\n"
         << "  \"label_count\": " << stats.label_count << ",\n"
         << "  \"pronunciation_count\": " << stats.pronunciation_count
         << ",\n"
         << "  \"pronunciation_token_count\": "
         << stats.pronunciation_token_count << ",\n"
         << "  \"multi_pronunciation_word_count\": "
         << stats.multi_pronunciation_word_count << ",\n"
         << "  \"homophone_path_count\": " << stats.homophone_path_count
         << ",\n"
         << "  \"dependency_hash_fnv1a64\": \""
         << asr_sdk::internal::flashlight_decoder::CompactLexiconHashHex(
                stats.dependency_hash)
         << "\",\n"
         << "  \"payload_hash_fnv1a64\": \""
         << asr_sdk::internal::flashlight_decoder::CompactLexiconHashHex(
                stats.payload_hash)
         << "\",\n"
         << "  \"file_size_bytes\": " << stats.file_size << "\n"
         << "}\n";
  if (!report) {
    throw std::runtime_error("failed to write report: " + path.string());
  }
}

CompactLexiconBuildStats Compile(const ModelPackage& package,
                                 const fs::path& output) {
  const auto resource = BuildLegacyResource(package);
  const uint64_t dependency_hash =
      asr_sdk::internal::flashlight_decoder::
          ComputeCompactLexiconDependencyHash(package);
  const auto stats =
      asr_sdk::internal::flashlight_decoder::WriteCompactLexicon(
          output, resource->LexiconTrie(),
          *resource->LegacyContextLookaheadForCompiler(),
          resource->BaseLexiconEntries(),
          static_cast<uint32_t>(resource->AmTokens().ModelVocabSize()),
          static_cast<uint32_t>(resource->OutputWords().Size()),
          resource->SilenceId(), resource->BlankId(),
          resource->UnknownWordId(), dependency_hash);
  asr_sdk::internal::flashlight_decoder::VerifyCompactLexiconAgainstLegacy(
      output, resource->LexiconTrie(),
      *resource->LegacyContextLookaheadForCompiler(),
      resource->BaseLexiconEntries(),
      static_cast<uint32_t>(resource->AmTokens().ModelVocabSize()),
      static_cast<uint32_t>(resource->OutputWords().Size()),
      resource->SilenceId(), resource->BlankId(), resource->UnknownWordId(),
      dependency_hash);
  return stats;
}

void Verify(const ModelPackage& package, const fs::path& input) {
  const auto resource = BuildLegacyResource(package);
  const uint64_t dependency_hash =
      asr_sdk::internal::flashlight_decoder::
          ComputeCompactLexiconDependencyHash(package);
  asr_sdk::internal::flashlight_decoder::VerifyCompactLexiconAgainstLegacy(
      input, resource->LexiconTrie(),
      *resource->LegacyContextLookaheadForCompiler(),
      resource->BaseLexiconEntries(),
      static_cast<uint32_t>(resource->AmTokens().ModelVocabSize()),
      static_cast<uint32_t>(resource->OutputWords().Size()),
      resource->SilenceId(), resource->BlankId(), resource->UnknownWordId(),
      dependency_hash);
}

void Inspect(const fs::path& input) {
  const auto lexicon = CompactLexicon::Load(input);
  std::cout << "format: compact_trie_v1\n"
            << "storage: "
            << (lexicon->IsMemoryMapped() ? "mmap" : "owned-buffer") << '\n'
            << "file_size_bytes: " << lexicon->FileSize() << '\n'
            << "tokens: " << lexicon->TokenCount() << '\n'
            << "words: " << lexicon->WordCount() << '\n'
            << "nodes: " << lexicon->NodeCount() << '\n'
            << "edges: " << lexicon->EdgeCount() << '\n'
            << "labels: " << lexicon->LabelCount() << '\n'
            << "pronunciations: " << lexicon->PronunciationCount() << '\n'
            << "pronunciation_tokens: "
            << lexicon->PronunciationTokenCount() << '\n'
            << "dependency_hash_fnv1a64: "
            << asr_sdk::internal::flashlight_decoder::CompactLexiconHashHex(
                   lexicon->DependencyHash())
            << '\n'
            << "payload_hash_fnv1a64: "
            << asr_sdk::internal::flashlight_decoder::CompactLexiconHashHex(
                   lexicon->PayloadHash())
            << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Arguments arguments = ParseArguments(argc, argv);
    if (arguments.command == "inspect") {
      Inspect(arguments.input);
      return 0;
    }
    const ModelPackage package = LoadSourcePackage(arguments.source_package);
    if (arguments.command == "compile") {
      const CompactLexiconBuildStats stats = Compile(package, arguments.output);
      WriteReport(arguments.report, arguments.source_package, arguments.output,
                  stats);
      std::cout << "compiled compact_trie_v1: " << arguments.output << " ("
                << stats.node_count << " nodes, " << stats.file_size
                << " bytes)\n";
    } else {
      Verify(package, arguments.input);
      std::cout << "verified compact_trie_v1: " << arguments.input << '\n';
    }
    return 0;
  } catch (const std::exception& error) {
    Usage(argv[0]);
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
