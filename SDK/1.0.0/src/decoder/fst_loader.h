#ifndef WENET_SDK_SRC_DECODER_FST_LOADER_H_
#define WENET_SDK_SRC_DECODER_FST_LOADER_H_

#include <filesystem>

#include "utils/status.h"

namespace wenet_sdk::internal {

struct FstStats {
  int64_t states = 0;
  int64_t arcs = 0;
  int64_t final_states = 0;
  int max_input_label = 0;
  int max_output_label = 0;
  bool openfst_validated = false;
};

Status ValidateFstFile(const std::filesystem::path& graph_path);
Status InspectFstFile(const std::filesystem::path& graph_path,
                      FstStats* stats);
Status ValidateFstCompatibility(const std::filesystem::path& graph_path,
                                int token_count, int word_count);

}  // namespace wenet_sdk::internal

#endif  // WENET_SDK_SRC_DECODER_FST_LOADER_H_
