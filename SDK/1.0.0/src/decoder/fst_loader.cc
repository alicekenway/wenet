#include "decoder/fst_loader.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

#ifdef WENETSDK_ENABLE_OPENFST
#include "fst/fstlib.h"  // NOLINT
#endif

#include "decoder/ctc_wfst_decoder.h"

namespace wenet_sdk::internal {

Status ValidateFstFile(const std::filesystem::path& graph_path) {
  return CtcWfstDecoder::ValidateGraphPath(graph_path);
}

Status InspectFstFile(const std::filesystem::path& graph_path,
                      FstStats* stats) {
  if (stats == nullptr) {
    return Status::InvalidArgument("FST stats output is null");
  }
  *stats = FstStats{};
  auto status = ValidateFstFile(graph_path);
  if (!status.ok()) {
    return status;
  }
#ifdef WENETSDK_ENABLE_OPENFST
  std::unique_ptr<fst::StdVectorFst> graph(fst::StdVectorFst::Read(graph_path.string()));
  if (!graph) {
    return Status::InvalidArgument("failed to read OpenFst graph: " +
                                   graph_path.string());
  }
  stats->openfst_validated = true;
  stats->states = graph->NumStates();
  for (fst::StateIterator<fst::StdVectorFst> siter(*graph); !siter.Done();
       siter.Next()) {
    const auto state = siter.Value();
    const auto final_weight = graph->Final(state);
    if (std::isfinite(final_weight.Value())) {
      ++stats->final_states;
    }
    for (fst::ArcIterator<fst::StdVectorFst> aiter(*graph, state);
         !aiter.Done(); aiter.Next()) {
      const auto& arc = aiter.Value();
      ++stats->arcs;
      stats->max_input_label =
          std::max(stats->max_input_label, static_cast<int>(arc.ilabel));
      stats->max_output_label =
          std::max(stats->max_output_label, static_cast<int>(arc.olabel));
    }
  }
#endif
  return Status::OK();
}

Status ValidateFstCompatibility(const std::filesystem::path& graph_path,
                                int token_count, int word_count) {
  FstStats stats;
  auto status = InspectFstFile(graph_path, &stats);
  if (!status.ok()) {
    return status;
  }
#ifdef WENETSDK_ENABLE_OPENFST
  if (stats.max_input_label > token_count) {
    return Status::InvalidArgument(
        "FST input labels exceed token table size: max ilabel " +
        std::to_string(stats.max_input_label) + ", token count " +
        std::to_string(token_count));
  }
  if (stats.max_output_label >= word_count) {
    return Status::InvalidArgument(
        "FST output labels exceed word table size: max olabel " +
        std::to_string(stats.max_output_label) + ", word count " +
        std::to_string(word_count));
  }
#else
  (void)token_count;
  (void)word_count;
#endif
  return Status::OK();
}

}  // namespace wenet_sdk::internal
