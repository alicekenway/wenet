# Decoder

Implemented decoder boundaries:

- `GreedyCtcDecoder`: argmax per frame, CTC collapse, blank removal.
- `CtcPrefixDecoder`: streaming CTC prefix beam search.
- `CtcWfstDecoder`: validates `TLG.fst` and exposes the WFST decoder boundary.
  With `WENETSDK_ENABLE_OPENFST=ON`, it loads an OpenFst `StdVectorFst` and runs
  streaming Viterbi token passing. Without OpenFst, it delegates to the prefix
  fallback so the SDK remains runnable.

The key production integration point is `src/decoder/ctc_wfst_decoder.*`.
The current backend is a compact in-SDK OpenFst search; a Kaldi-style online
lattice decoder can be added later if full lattice/N-best semantics are needed.
