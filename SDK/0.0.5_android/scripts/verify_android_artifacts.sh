#!/usr/bin/env bash
set -euo pipefail

SDK_ROOT="${SDK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ROOT="${ROOT:-$(cd "${SDK_ROOT}/../../.." && pwd)}"
TEST_ROOT="${TEST_ROOT:-${ROOT}/test/0.0.5_android}"
APK_PATH="${APK_PATH:-${SDK_ROOT}/android_demo/app/out/manual-x86_64/app-debug.apk}"
OUT_DIR="${OUT_DIR:-${TEST_ROOT}/android_build_verification}"
ANDROID_NDK="${ANDROID_NDK:-/tmp/asr_android_toolchain/android-ndk-r26d}"
ANDROID_HOME="${ANDROID_HOME:-/tmp/asr_android_toolchain/android-sdk}"
JAVA_HOME="${JAVA_HOME:-/tmp/asr_android_toolchain/jdk-17.0.19+10}"

mkdir -p "${OUT_DIR}"
REPORT="${OUT_DIR}/verification_report.txt"
: > "${REPORT}"

log() {
  printf '%s\n' "$*" | tee -a "${REPORT}"
}

fail() {
  log "FAIL: $*"
  exit 1
}

require_file() {
  local path="$1"
  [ -f "${path}" ] || fail "missing file: ${path}"
}

READELF="${READELF:-}"
if [ -z "${READELF}" ] && [ -x "${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf" ]; then
  READELF="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-readelf"
fi
if [ -z "${READELF}" ]; then
  READELF="$(command -v readelf || true)"
fi
[ -n "${READELF}" ] || fail "readelf/llvm-readelf not found"

X86_LIB="${SDK_ROOT}/out-sdk-android-x86_64/libasr_sdk.so"
ARM64_LIB="${SDK_ROOT}/out-sdk-android-arm64-v8a/libasr_sdk.so"
HOST_OUTPUT="${TEST_ROOT}/reduced_greedy_host_test/output.txt"

require_file "${X86_LIB}"
require_file "${ARM64_LIB}"
require_file "${APK_PATH}"
require_file "${HOST_OUTPUT}"

log "Verifying Android SDK libraries"
"${READELF}" -h "${X86_LIB}" > "${OUT_DIR}/x86_64_libasr_sdk.elf.txt"
"${READELF}" -h "${ARM64_LIB}" > "${OUT_DIR}/arm64_v8a_libasr_sdk.elf.txt"
grep -q "Machine:.*X86-64" "${OUT_DIR}/x86_64_libasr_sdk.elf.txt" \
  || fail "x86_64 libasr_sdk.so machine type is not X86-64"
grep -q "Machine:.*AArch64" "${OUT_DIR}/arm64_v8a_libasr_sdk.elf.txt" \
  || fail "arm64-v8a libasr_sdk.so machine type is not AArch64"
log "OK: x86_64 and arm64-v8a libasr_sdk.so machine types"

"${READELF}" -d "${X86_LIB}" > "${OUT_DIR}/x86_64_libasr_sdk.dynamic.txt"
for needed in libonnxruntime.so libc++_shared.so liblog.so libc.so; do
  grep -q "${needed}" "${OUT_DIR}/x86_64_libasr_sdk.dynamic.txt" \
    || fail "x86_64 libasr_sdk.so missing NEEDED ${needed}"
done
log "OK: x86_64 libasr_sdk.so dynamic dependencies"

log "Verifying APK contents"
unzip -l "${APK_PATH}" > "${OUT_DIR}/apk_contents.txt"
for entry in \
  "assets/model/model.onnx" \
  "assets/model/sdk_model.json" \
  "assets/model/tokens.txt" \
  "assets/test.wav" \
  "classes.dex" \
  "lib/x86_64/libasr_sdk.so" \
  "lib/x86_64/libasr_jni.so" \
  "lib/x86_64/libonnxruntime.so" \
  "lib/x86_64/libc++_shared.so"; do
  grep -q "${entry}" "${OUT_DIR}/apk_contents.txt" \
    || fail "APK missing ${entry}"
done
log "OK: APK required contents"

APKSIGNER="${ANDROID_HOME}/build-tools/35.0.0/apksigner"
if [ -x "${APKSIGNER}" ] && [ -x "${JAVA_HOME}/bin/java" ]; then
  log "Verifying APK signature"
  PATH="${JAVA_HOME}/bin:${PATH}" "${APKSIGNER}" verify --verbose "${APK_PATH}" \
    > "${OUT_DIR}/apk_signature.txt"
  grep -q "Verifies" "${OUT_DIR}/apk_signature.txt" \
    || fail "APK signature verification did not report Verifies"
  grep -q "Verified using v2 scheme.*true" "${OUT_DIR}/apk_signature.txt" \
    || fail "APK v2 signature verification failed"
  log "OK: APK signature"
else
  log "SKIP: apksigner or Java not available for signature verification"
fi

log "Verifying host model test output"
grep -q "^000000003 " "${HOST_OUTPUT}" \
  || fail "host output missing utterance 000000003"
grep -q "^000000007 " "${HOST_OUTPUT}" \
  || fail "host output missing utterance 000000007"
log "OK: host reduced greedy output includes expected first five utterance range"

log "All non-runtime Android artifact checks passed."
