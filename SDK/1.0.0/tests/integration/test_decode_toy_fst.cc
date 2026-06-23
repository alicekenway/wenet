#include <cassert>
#include <cmath>
#include <filesystem>
#include <vector>

#include "decoder/ctc_wfst_decoder.h"
#include "decoder/fst_loader.h"

#ifdef WENETSDK_ENABLE_OPENFST
#include "fst/fstlib.h"  // NOLINT
#endif

int main() {
#ifndef WENETSDK_ENABLE_OPENFST
  return 0;
#else
  const auto graph_path =
      std::filesystem::temp_directory_path() / "wenet_sdk_decode_toy.fst";
  fst::StdVectorFst graph;
  const auto s0 = graph.AddState();
  const auto s1 = graph.AddState();
  const auto s2 = graph.AddState();
  const auto s3 = graph.AddState();
  graph.SetStart(s0);
  graph.SetFinal(s3, fst::StdArc::Weight::One());
  graph.AddArc(s0, fst::StdArc(2, 1, 0.0f, s1));
  graph.AddArc(s1, fst::StdArc(0, 0, 0.0f, s2));
  graph.AddArc(s2, fst::StdArc(3, 2, 0.0f, s3));
  assert(graph.Write(graph_path.string()));

  wenet_sdk::internal::FstStats stats;
  auto status = wenet_sdk::internal::InspectFstFile(graph_path, &stats);
  assert(status.ok());
  assert(stats.openfst_validated);
  assert(stats.states == 4);
  assert(stats.arcs == 3);
  status = wenet_sdk::internal::ValidateFstCompatibility(graph_path, 3, 3);
  assert(status.ok());

  wenet_sdk::internal::CtcWfstDecoderOptions options;
  options.blank_id = 0;
  options.beam = 20.0f;
  wenet_sdk::internal::CtcWfstDecoder decoder(options, graph_path);
  const std::vector<std::vector<float>> logp = {
      {std::log(0.01f), std::log(0.98f), std::log(0.01f)},
      {std::log(0.01f), std::log(0.01f), std::log(0.98f)}};
  decoder.Advance(logp);
  const auto result = decoder.Finalize();
  assert((result.word_ids == std::vector<int>{1, 2}));
  assert((result.token_ids == std::vector<int>{1, 2}));
  std::filesystem::remove(graph_path);
  return 0;
#endif
}
