#!/usr/bin/env bash
set -euo pipefail

VERSION="${VERSION:-0.0.5}"

SDK_ROOT="${SDK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "${SDK_ROOT}/../../.." && pwd)}"
MODEL_PACKAGE="${MODEL_PACKAGE:-${REPO_ROOT}/test/0.0.5/sherpa-onnx-en-wenet-gigaspeech_int8_control_ft/package}"
DIST_DIR="${DIST_DIR:-${SDK_ROOT}/dist}"
OUTPUT_ZIP="${OUTPUT_ZIP:-${DIST_DIR}/asr-model-${VERSION}.zip}"

required_files=(
  sdk_model.json
  model.onnx
  tokens.txt
  words.txt
  lexicon.txt
  lm.bin
  output_mapping.txt
  final_output_mapping.txt
)

: "${MODEL_PACKAGE:?MODEL_PACKAGE is required}"

if [ ! -d "${MODEL_PACKAGE}" ]; then
  echo "MODEL_PACKAGE does not exist: ${MODEL_PACKAGE}" >&2
  exit 1
fi

for file in "${required_files[@]}"; do
  if [ ! -f "${MODEL_PACKAGE}/${file}" ]; then
    echo "Missing required model file: ${MODEL_PACKAGE}/${file}" >&2
    exit 1
  fi
done

mkdir -p "${DIST_DIR}"
rm -f "${OUTPUT_ZIP}"
(
  cd "${MODEL_PACKAGE}"
  zip -qr "${OUTPUT_ZIP}" .
)

unzip -l "${OUTPUT_ZIP}" | grep 'sdk_model.json' >/dev/null
echo "${OUTPUT_ZIP}"
