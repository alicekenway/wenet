#include <cassert>
#include <cmath>
#include <filesystem>
#include <vector>

#include "decoder/ctc_wfst_decoder.h"

#ifdef WENETSDK_ENABLE_OPENFST
#include "fst/fstlib.h"  // NOLINT
#endif

int main() {
#ifndef WENETSDK_ENABLE_OPENFST
  return 0;
#else
  const std::filesystem::path graph_path =
      std::filesystem::temp_directory_path() / "wenet_sdk_toy_tlg.fst";

  fst::StdVectorFst fst;
  const auto s0 = fst.AddState();
  const auto s1 = fst.AddState();
  const auto s2 = fst.AddState();
  fst.SetStart(s0);
  fst.SetFinal(s2, fst::StdArc::Weight::One());
  fst.AddArc(s0, fst::StdArc(2, 1, 0.0f, s1));  // token id 1 -> word id 1.
  fst.AddArc(s1, fst::StdArc(3, 2, 0.0f, s2));  // token id 2 -> word id 2.
  assert(fst.Write(graph_path.string()));

  auto status = wenet_sdk::internal::CtcWfstDecoder::ValidateGraphPath(graph_path);
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
  const std::vector<int> words = {1, 2};
  const std::vector<int> tokens = {1, 2};
  assert(result.word_ids == words);
  assert(result.token_ids == tokens);
  std::filesystem::remove(graph_path);
  return 0;
#endif
}
