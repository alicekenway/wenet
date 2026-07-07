#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 AM_ONNX_DIR TLG_LANG_DIR OUT_MODEL_DIR" >&2
  exit 2
fi

am_dir=$1
tlg_dir=$2
out_dir=$3

mkdir -p "${out_dir}"

ln -sfn "${am_dir}/encoder.onnx" "${out_dir}/encoder.onnx"
ln -sfn "${am_dir}/ctc.onnx" "${out_dir}/ctc.onnx"
ln -sfn "${am_dir}/decoder.onnx" "${out_dir}/decoder.onnx"
ln -sfn "${tlg_dir}/units.txt" "${out_dir}/units.txt"
ln -sfn "${tlg_dir}/words.txt" "${out_dir}/words.txt"
ln -sfn "${tlg_dir}/TLG.fst" "${out_dir}/TLG.fst"

cat > "${out_dir}/sdk_model.json" <<JSON
{
  "sdk_model_version": 1,
  "backend": "wenet_onnxruntime_static_wenet_dynamic_ort",
  "audio": {
    "sample_rate": 16000
  },
  "wenet": {
    "onnx_dir": ".",
    "unit_path": "units.txt",
    "dict_path": "words.txt",
    "fst_path": "TLG.fst"
  },
  "decode": {
    "chunk_size": 16,
    "num_left_chunks": 16,
    "nbest": 1
  },
  "runtime": {
    "enable_continuous_decoding": true
  },
  "postprocess": {
    "language_type": "chs",
    "enable_timestamp": false
  }
}
JSON

echo "${out_dir}"
