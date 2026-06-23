#include <filesystem>
#include <iostream>
#include <string>

#include "decoder/fst_loader.h"
#include "model/model_metadata.h"

namespace {

std::string ArgValue(int argc, char** argv, const std::string& name,
                     const std::string& fallback = "") {
  for (int i = 1; i + 1 < argc; ++i) {
    if (argv[i] == name) {
      return argv[i + 1];
    }
  }
  return fallback;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model_dir = ArgValue(argc, argv, "--model_dir");
  const std::string graph = ArgValue(argc, argv, "--graph");
  std::filesystem::path graph_path;
  if (!graph.empty()) {
    graph_path = graph;
  } else if (!model_dir.empty()) {
    wenet_sdk::internal::ModelMetadata metadata;
    auto status = wenet_sdk::internal::LoadModelMetadata(model_dir, &metadata);
    if (!status.ok()) {
      std::cerr << status.message() << "\n";
      return 1;
    }
    graph_path = metadata.Resolve(metadata.decoder.graph);
  } else {
    std::cerr << "usage: inspect_fst --model_dir MODEL_DIR | --graph TLG.fst\n";
    return 2;
  }

  wenet_sdk::internal::FstStats stats;
  auto status = wenet_sdk::internal::InspectFstFile(graph_path, &stats);
  if (!status.ok()) {
    std::cerr << status.message() << "\n";
    return 1;
  }
  std::cout << "graph: " << graph_path.string() << "\n";
  std::cout << "size_bytes: " << std::filesystem::file_size(graph_path)
            << "\n";
  if (stats.openfst_validated) {
    std::cout << "states: " << stats.states << "\n";
    std::cout << "arcs: " << stats.arcs << "\n";
    std::cout << "final_states: " << stats.final_states << "\n";
    std::cout << "max_input_label: " << stats.max_input_label << "\n";
    std::cout << "max_output_label: " << stats.max_output_label << "\n";
  }
  std::cout << "openfst_backend: "
#ifdef WENETSDK_ENABLE_OPENFST
            << "enabled";
#else
            << "stub-validation-only";
#endif
  std::cout << "\n";
  return 0;
}
