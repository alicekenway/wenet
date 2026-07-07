#!/usr/bin/env bash
set -euo pipefail

ABI="${1:-x86_64}"

SDK_ROOT="${SDK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ANDROID_DEMO_ROOT="${ANDROID_DEMO_ROOT:-${SDK_ROOT}/android_demo}"
ORT_ANDROID_ROOT="${ORT_ANDROID_ROOT:-${SDK_ROOT}/third_party/onnxruntime-android}"

: "${ORT_ANDROID_ROOT:?ORT_ANDROID_ROOT is required}"

OUT_DIR="${ANDROID_DEMO_ROOT}/app/src/main/jniLibs/${ABI}"
mkdir -p "${OUT_DIR}"

SDK_OUTPUT_DIR="${SDK_OUTPUT_DIR:-${SDK_ROOT}/out-sdk-android-${ABI}}"

cp "${SDK_OUTPUT_DIR}/libasr_sdk.so" "${OUT_DIR}/"

if [ -f "${ORT_ANDROID_ROOT}/jni/${ABI}/libonnxruntime.so" ]; then
  cp "${ORT_ANDROID_ROOT}/jni/${ABI}/libonnxruntime.so" "${OUT_DIR}/"
elif [ -f "${ORT_ANDROID_ROOT}/lib/${ABI}/libonnxruntime.so" ]; then
  cp "${ORT_ANDROID_ROOT}/lib/${ABI}/libonnxruntime.so" "${OUT_DIR}/"
else
  echo "Cannot find ONNX Runtime Android library for ${ABI}" >&2
  exit 1
fi

find "${OUT_DIR}" -maxdepth 1 -type f -print | sort
