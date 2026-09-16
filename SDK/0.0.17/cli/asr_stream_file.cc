#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "asr_sdk/asr_engine.h"
#include "asr_sdk/decode_context.h"
#include "audio/wav_reader.h"
#include "utils/timer.h"
#include "utils/slot_file.h"

namespace {

struct Args {
  std::string model_dir;
  std::string wav;
  int chunk_ms = 100;
  bool print_partial = true;
  bool debug = false;
  std::string contacts_tsv;
  std::string slots;
  bool skip_unencodable_contacts = false;
};

bool ParseBool(const std::string& value) {
  return value == "1" || value == "true" || value == "yes" ||
         value == "on";
}

void Usage(const char* argv0) {
  std::cerr << "usage: " << argv0
            << " --model_dir DIR --wav WAV [--chunk_ms 100]"
               " [--print_partial true] [--debug false]"
               " [--contacts_tsv PATH] [--slots PATH]"
               " [--skip_unencodable_contacts false]\n";
}

bool ParseArgs(int argc, char** argv, Args* args) {
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto need_value = [&](std::string* out) -> bool {
      if (i + 1 >= argc) {
        return false;
      }
      *out = argv[++i];
      return true;
    };
    if (key == "--model_dir") {
      if (!need_value(&args->model_dir)) return false;
    } else if (key == "--wav") {
      if (!need_value(&args->wav)) return false;
    } else if (key == "--chunk_ms") {
      std::string value;
      if (!need_value(&value)) return false;
      args->chunk_ms = std::max(1, std::atoi(value.c_str()));
    } else if (key == "--print_partial") {
      std::string value;
      if (!need_value(&value)) return false;
      args->print_partial = ParseBool(value);
    } else if (key == "--debug") {
      std::string value;
      if (!need_value(&value)) return false;
      args->debug = ParseBool(value);
    } else if (key == "--contacts_tsv") {
      if (!need_value(&args->contacts_tsv)) return false;
    } else if (key == "--slots") {
      if (!need_value(&args->slots)) return false;
    } else if (key == "--skip_unencodable_contacts") {
      std::string value;
      if (!need_value(&value)) return false;
      args->skip_unencodable_contacts = ParseBool(value);
    } else if (key == "--help" || key == "-h") {
      return false;
    } else {
      std::cerr << "unknown argument: " << key << "\n";
      return false;
    }
  }
  return !args->model_dir.empty() && !args->wav.empty();
}

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

std::vector<std::string> SplitTabs(const std::string& line) {
  std::vector<std::string> fields;
  size_t begin = 0;
  while (true) {
    const size_t tab = line.find('\t', begin);
    fields.push_back(line.substr(begin, tab == std::string::npos
                                            ? std::string::npos
                                            : tab - begin));
    if (tab == std::string::npos) {
      break;
    }
    begin = tab + 1;
  }
  return fields;
}

std::vector<std::string> SplitWhitespace(const std::string& value) {
  std::istringstream input(value);
  std::vector<std::string> output;
  std::string item;
  while (input >> item) {
    output.push_back(item);
  }
  return output;
}

asr_sdk::Status LoadContactsTsv(const std::string& path,
                                asr_sdk::DecodeContextConfig* config) {
  std::ifstream input(path);
  if (!input) {
    return asr_sdk::Status::NotFound("failed to open contacts TSV: " + path);
  }
  std::unordered_map<std::string, size_t> contact_index;
  std::unordered_set<std::string> seen_rows;
  std::string line;
  int line_no = 0;
  while (std::getline(input, line)) {
    ++line_no;
    const std::string trimmed = Trim(line);
    if (trimmed.empty() || trimmed.front() == '#') {
      continue;
    }
    std::vector<std::string> fields = SplitTabs(line);
    if (fields.size() < 3 || fields.size() > 5) {
      return asr_sdk::Status::InvalidArgument(
          path + ":" + std::to_string(line_no) +
          ": expected 3 to 5 tab-separated fields");
    }
    const std::string id = Trim(fields[0]);
    const std::string display_name = Trim(fields[1]);
    const std::string spoken_form = Trim(fields[2]);
    if (id.empty() || display_name.empty() || spoken_form.empty()) {
      return asr_sdk::Status::InvalidArgument(
          path + ":" + std::to_string(line_no) +
          ": contact_id, display_name, and spoken_form are required");
    }

    asr_sdk::ContactSpokenForm form;
    form.text = spoken_form;
    if (fields.size() >= 4 && !Trim(fields[3]).empty()) {
      form.am_tokens = SplitWhitespace(fields[3]);
    }
    if (fields.size() == 5 && !Trim(fields[4]).empty()) {
      try {
        size_t consumed = 0;
        form.logical_word_count = std::stoi(Trim(fields[4]), &consumed);
        if (consumed != Trim(fields[4]).size() ||
            form.logical_word_count <= 0) {
          throw std::invalid_argument("invalid count");
        }
      } catch (...) {
        return asr_sdk::Status::InvalidArgument(
            path + ":" + std::to_string(line_no) +
            ": logical_word_count must be a positive integer");
      }
    }
    const std::string row_key = id + "\x1e" + display_name + "\x1e" +
                                spoken_form + "\x1e" + Trim(fields.size() >= 4
                                                                  ? fields[3]
                                                                  : "") +
                                "\x1e" + std::to_string(form.logical_word_count);
    if (!seen_rows.insert(row_key).second) {
      continue;
    }
    const auto existing = contact_index.find(id);
    if (existing == contact_index.end()) {
      contact_index[id] = config->contacts.size();
      config->contacts.push_back(asr_sdk::ContactEntry{});
      config->contacts.back().contact_id = id;
      config->contacts.back().display_name = display_name;
      config->contacts.back().spoken_forms.push_back(std::move(form));
    } else {
      asr_sdk::ContactEntry& entry = config->contacts[existing->second];
      if (entry.display_name != display_name) {
        return asr_sdk::Status::InvalidArgument(
            path + ":" + std::to_string(line_no) +
            ": repeated contact_id has a conflicting display_name");
      }
      entry.spoken_forms.push_back(std::move(form));
    }
  }
  return asr_sdk::Status::Ok();
}

}  // namespace

int main(int argc, char** argv) {
  Args args;
  if (!ParseArgs(argc, argv, &args)) {
    Usage(argv[0]);
    return 2;
  }

  asr_sdk::internal::WavData wav;
  asr_sdk::Status status = asr_sdk::internal::ReadWavFile(args.wav, &wav);
  if (!status.ok()) {
    std::cerr << status.ToString() << "\n";
    return 1;
  }

  asr_sdk::EngineConfig config;
  config.model_dir = args.model_dir;
  config.sample_rate = wav.sample_rate;
  config.debug = args.debug;
  auto engine_or = asr_sdk::AsrEngine::Create(config);
  if (!engine_or.ok()) {
    std::cerr << engine_or.status().ToString() << "\n";
    return 1;
  }
  auto engine = std::move(engine_or).value();
  std::shared_ptr<const asr_sdk::DecodeContext> context;
  if (!args.contacts_tsv.empty() || !args.slots.empty()) {
    asr_sdk::DecodeContextConfig context_config;
    status = args.slots.empty()
                 ? LoadContactsTsv(args.contacts_tsv, &context_config)
                 : asr_sdk::internal::LoadSlotSectionFile(args.slots,
                                                          &context_config);
    if (!status.ok()) {
      std::cerr << status.ToString() << "\n";
      return 1;
    }
    context_config.contact_list.skip_unencodable_forms =
        args.skip_unencodable_contacts;
    auto context_or = engine->CompileDecodeContext(context_config);
    if (!context_or.ok()) {
      std::cerr << context_or.status().ToString() << "\n";
      return 1;
    }
    context = std::move(context_or).value();
  }
  auto stream_or = context ? engine->CreateStream(context) : engine->CreateStream();
  if (!stream_or.ok()) {
    std::cerr << stream_or.status().ToString() << "\n";
    return 1;
  }
  auto stream = std::move(stream_or).value();

  const size_t chunk_samples =
      static_cast<size_t>(wav.sample_rate * args.chunk_ms / 1000);
  asr_sdk::internal::Timer timer;
  std::string last_partial;
  for (size_t offset = 0; offset < wav.samples.size();
       offset += chunk_samples) {
    const size_t n = std::min(chunk_samples, wav.samples.size() - offset);
    status = stream->AcceptPcm16(wav.samples.data() + offset, n,
                                 wav.sample_rate);
    if (!status.ok()) {
      std::cerr << status.ToString() << "\n";
      return 1;
    }
    while (stream->DecodeReady()) {
      status = stream->Decode();
      if (!status.ok()) {
        std::cerr << status.ToString() << "\n";
        return 1;
      }
      const auto partial = stream->GetResult();
      if (args.print_partial && !partial.text.empty() &&
          partial.text != last_partial && !partial.is_final) {
        last_partial = partial.text;
        std::cout << "[partial] " << partial.text << "\n";
      }
    }
  }

  status = stream->SetInputFinished();
  if (!status.ok()) {
    std::cerr << status.ToString() << "\n";
    return 1;
  }
  while (stream->DecodeReady()) {
    status = stream->Decode();
    if (!status.ok()) {
      std::cerr << status.ToString() << "\n";
      return 1;
    }
  }
  const auto final_result = stream->GetFinalResult();
  const double wall_sec = timer.ElapsedSeconds();
  const double audio_sec =
      static_cast<double>(wav.samples.size()) / wav.sample_rate;
  std::cout << "[final] " << final_result.text << "\n";
  for (const asr_sdk::EntityResult& entity : final_result.entities) {
    if (entity.type != "contact") {
      continue;
    }
    std::cout << "[contact] " << entity.text << " score=" << entity.score
              << " ambiguous=" << (entity.ambiguous ? "true" : "false")
              << " candidates=";
    for (size_t i = 0; i < entity.candidates.size(); ++i) {
      if (i != 0) {
        std::cout << ",";
      }
      std::cout << entity.candidates[i].contact_id << ":"
                << entity.candidates[i].display_name;
    }
    std::cout << "\n";
  }
  if (args.debug && !final_result.raw_backend_json.empty()) {
    std::cerr << final_result.raw_backend_json << "\n";
  }
  std::cout << "audio_sec: " << audio_sec << "\n";
  std::cout << "wall_sec: " << wall_sec << "\n";
  std::cout << "RTF: " << (wall_sec / audio_sec) << "\n";
  std::cout << "wenet_linkage: static\n";
  std::cout << "onnxruntime_linkage: dynamic\n";
  return 0;
}
