#!/usr/bin/env bash
set -euo pipefail

SDK_ROOT="${SDK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "${SDK_ROOT}/../../.." && pwd)}"
ANDROID_DEMO_ROOT="${ANDROID_DEMO_ROOT:-${SDK_ROOT}/android_demo}"
MODEL_PACKAGE="${MODEL_PACKAGE:-${REPO_ROOT}/test/0.0.5/sherpa-onnx-en-wenet-gigaspeech_int8_control_ft/package}"
TEST_WAV="${TEST_WAV:-${REPO_ROOT}/data/ENX/test_ONEASR-2061.utf8.part/wav/000000003.wav}"

: "${MODEL_PACKAGE:?MODEL_PACKAGE is required}"
: "${TEST_WAV:?TEST_WAV is required}"

ASSETS_DIR="${ANDROID_DEMO_ROOT}/app/src/main/assets"
MODEL_ASSETS_DIR="${ASSETS_DIR}/model"
mkdir -p "${MODEL_ASSETS_DIR}"

rm -rf "${MODEL_ASSETS_DIR:?}"/*
cp -a "${MODEL_PACKAGE}/." "${MODEL_ASSETS_DIR}/"
cp "${TEST_WAV}" "${ASSETS_DIR}/test.wav"
if [ -f "${MODEL_ASSETS_DIR}/checksums.sha256" ]; then
  sed -i -E 's#  .*/([^/]+)$#  \1#' "${MODEL_ASSETS_DIR}/checksums.sha256"
fi

find "${ASSETS_DIR}" -maxdepth 2 -type f -printf '%P %s\n' | sort
