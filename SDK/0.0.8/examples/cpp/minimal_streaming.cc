#include <algorithm>
#include <iostream>

#include "asr_sdk/asr_engine.h"

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " MODEL_DIR\n";
    return 2;
  }
  asr_sdk::EngineConfig config;
  config.model_dir = argv[1];
  auto engine_or = asr_sdk::AsrEngine::Create(config);
  if (!engine_or.ok()) {
    std::cerr << engine_or.status().ToString() << "\n";
    return 1;
  }
  std::cout << "engine created for " << config.model_dir << "\n";
  return 0;
}
