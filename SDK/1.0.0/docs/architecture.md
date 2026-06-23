# Architecture

The SDK follows the plan's ownership split:

- `AsrEngine` owns immutable model package metadata, token/word symbol tables,
  and configuration.
- `StreamSession` owns one mutable streaming recognizer.
- `Recognizer` orchestrates feature extraction, model forward, blank skipping,
  streaming decoding, endpointing, and result building.
- `FeaturePipeline` accepts float32 or int16 PCM, resamples mismatched input
  sample rates to the model package sample rate, then performs frame extraction,
  fbank, and optional CMVN.

Public headers under `include/wenet_sdk` expose no ONNX Runtime, OpenFst, Kaldi,
YAML, or JSON types. Internal implementations live under `src`.

The default model backend is a deterministic ONNX stub so the SDK is buildable in
minimal environments. The `OnnxCtcModel` class is the integration boundary for a
real ONNX Runtime implementation.
