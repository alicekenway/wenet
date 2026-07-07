#!/usr/bin/env bash
set -euo pipefail

ABI="${1:-x86_64}"
MINSDK="${MINSDK:-26}"

: "${ANDROID_HOME:?ANDROID_HOME is required}"
: "${ANDROID_NDK:?ANDROID_NDK is required}"
: "${WENET_ROOT:?WENET_ROOT is required}"

if [ "${ASR_SDK_TRY_FULL_WENET_ANDROID_RUNTIME:-0}" != "1" ]; then
  cat >&2 <<'EOF'
The current WeNet runtime Android CMake path does not make the SDK's
OnnxAsrModel bridge directly. Use the default reduced Android SDK path:

  ./scripts/stage_android_greedy_package.sh
  ./scripts/make_sdk_android.sh x86_64

Set ASR_SDK_TRY_FULL_WENET_ANDROID_RUNTIME=1 only for experimental full
runtime porting work.
EOF
  exit 2
fi

RUNTIME_ROOT="${WENET_ROOT}/runtime/onnxruntime"
OUTPUT_DIR="${RUNTIME_ROOT}/out-android-${ABI}"

cmake -S "${RUNTIME_ROOT}" -B "${OUTPUT_DIR}" \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ABI}" \
  -DANDROID_PLATFORM="android-${MINSDK}" \
  -DANDROID_STL=c++_shared \
  -DCMAKE_BUILD_TYPE=Release \
  -DTORCH=OFF \
  -DONNX=ON \
  -DWEBSOCKET=OFF \
  -DGRPC=OFF \
  -DHTTP=OFF \
  -DGRAPH_TOOLS=OFF \
  -DBUILD_TESTING=OFF

cmake --build "${OUTPUT_DIR}" -j"$(nproc)"

test -f "${OUTPUT_DIR}/decoder/libdecoder.a"
file "${OUTPUT_DIR}/decoder/libdecoder.a" || true
