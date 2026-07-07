#!/usr/bin/env bash
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.25.1}"
SDK_ROOT="${SDK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR="${OUT_DIR:-${SDK_ROOT}/third_party/onnxruntime-android}"

: "${AAR_PATH:?Please set AAR_PATH=/path/to/onnxruntime-android-${ORT_VERSION}.aar}"

rm -rf /tmp/asr-sdk-onnxruntime-android-aar
mkdir -p /tmp/asr-sdk-onnxruntime-android-aar "${OUT_DIR}"
unzip -q "${AAR_PATH}" -d /tmp/asr-sdk-onnxruntime-android-aar

rm -rf "${OUT_DIR}/headers" "${OUT_DIR}/jni"
cp -a /tmp/asr-sdk-onnxruntime-android-aar/headers "${OUT_DIR}/"
cp -a /tmp/asr-sdk-onnxruntime-android-aar/jni "${OUT_DIR}/"

find "${OUT_DIR}" -maxdepth 3 -type f | sort
