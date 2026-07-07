#!/usr/bin/env bash
set -euo pipefail

SDK_ROOT="${SDK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "${SDK_ROOT}/../../.." && pwd)}"
SOURCE_PACKAGE="${SOURCE_PACKAGE:-${REPO_ROOT}/test/0.0.5/sherpa-onnx-en-wenet-gigaspeech_int8_control_ft/package}"
OUT_PACKAGE="${OUT_PACKAGE:-${REPO_ROOT}/test/0.0.5_android/greedy_package}"

: "${SOURCE_PACKAGE:?SOURCE_PACKAGE is required}"
: "${OUT_PACKAGE:?OUT_PACKAGE is required}"

mkdir -p "${OUT_PACKAGE}"
cp "${SOURCE_PACKAGE}/model.onnx" "${OUT_PACKAGE}/model.onnx"
cp "${SOURCE_PACKAGE}/tokens.txt" "${OUT_PACKAGE}/tokens.txt"

cat > "${OUT_PACKAGE}/sdk_model.json" <<'JSON'
{
  "decoder_type": "ctc_greedy",
  "model_path": "model.onnx",
  "tokens": "tokens.txt",
  "feature_type": "kaldi",
  "blank_token": "<blank>",
  "sample_rate": 16000,
  "debug": false
}
JSON

find "${OUT_PACKAGE}" -maxdepth 1 -type f -printf '%f %s\n' | sort
