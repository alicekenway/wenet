# WeNet Static + Dynamic ONNX Runtime SDK

This SDK exposes a small C++ and C ABI wrapper around the forked WeNet ONNX
runtime. Public headers live under `include/asr_sdk` and do not include WeNet or
ONNX Runtime headers.

The build links WeNet runtime archives privately into `libasr_sdk.so` and keeps
ONNX Runtime as a dynamic dependency. The default paths are set for this local
workspace:

```bash
cmake -S . -B build
cmake --build build -j
```

Prepare a flat WeNet runtime package from the supplied AM and TLG directories:

```bash
scripts/prepare_runtime_package.sh \
  /home/jinyang_wang/Dev/ASR/ASR_wenet/model/wenet_efficient_conformer_aishell_v2_onnx \
  /home/jinyang_wang/Dev/ASR/ASR_wenet/LM/wenet_lm/tlg/lang_test \
  /home/jinyang_wang/Dev/ASR/ASR_wenet/test/0.0.1/model_wfst
```

Run a WAV through the SDK:

```bash
build/asr_stream_file \
  --model_dir /home/jinyang_wang/Dev/ASR/ASR_wenet/test/0.0.1/model_wfst \
  --wav /home/jinyang_wang/Dev/ASR/ASR_wenet/data/hf_wenetspeech_test_net/wenetspeech_test_net_sample_2000/wav/000000037.wav
```
