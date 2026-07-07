#!/usr/bin/env bash
set -euo pipefail

SDK_ROOT="${SDK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "${SDK_ROOT}/../../.." && pwd)}"
APK_PATH="${APK_PATH:-${SDK_ROOT}/android_demo/app/out/manual-x86_64/app-debug.apk}"
PACKAGE_NAME="${PACKAGE_NAME:-com.example.asrdemo}"
ACTIVITY_NAME="${ACTIVITY_NAME:-.MainActivity}"
LOG_TAG="${LOG_TAG:-ASR_TEST}"
OUT_DIR="${OUT_DIR:-${REPO_ROOT}/test/0.0.5_android/android_runtime_test}"
LOGCAT_SECONDS="${LOGCAT_SECONDS:-120}"
RESULT_MARKER="${RESULT_MARKER:-final_text=}"

ADB_BIN="${ADB:-adb}"
ADB_CMD=("${ADB_BIN}")
if [ -n "${ADB_SERIAL:-}" ]; then
  ADB_CMD+=("-s" "${ADB_SERIAL}")
fi

mkdir -p "${OUT_DIR}"

if [ ! -f "${APK_PATH}" ]; then
  echo "APK not found: ${APK_PATH}" >&2
  exit 2
fi

"${ADB_BIN}" start-server >/dev/null

if ! "${ADB_CMD[@]}" get-state >/dev/null 2>&1; then
  cat >&2 <<EOF
No Android device/emulator is attached.

Expected one of:
  adb connect localhost:5555
  adb devices

Then rerun:
  ADB_SERIAL="${ADB_SERIAL:-localhost:15555}" APK_PATH="${APK_PATH}" OUT_DIR="${OUT_DIR}" ${0}
EOF
  exit 3
fi

"${ADB_BIN}" devices > "${OUT_DIR}/adb_devices.txt"
"${ADB_CMD[@]}" install -r "${APK_PATH}" | tee "${OUT_DIR}/adb_install.txt"
"${ADB_CMD[@]}" logcat -c
"${ADB_CMD[@]}" shell am start -n "${PACKAGE_NAME}/${ACTIVITY_NAME}" | tee "${OUT_DIR}/adb_start.txt"

deadline=$((SECONDS + LOGCAT_SECONDS))
while [ "${SECONDS}" -lt "${deadline}" ]; do
  "${ADB_CMD[@]}" logcat -d -v time -s "${LOG_TAG}" \
    > "${OUT_DIR}/logcat_${LOG_TAG}.txt"
  if grep -q "${RESULT_MARKER}" "${OUT_DIR}/logcat_${LOG_TAG}.txt"; then
    echo "Runtime ASR log found: ${OUT_DIR}/logcat_${LOG_TAG}.txt"
    exit 0
  fi
  sleep 2
done

if grep -q "${RESULT_MARKER}" "${OUT_DIR}/logcat_${LOG_TAG}.txt"; then
  echo "Runtime ASR log found: ${OUT_DIR}/logcat_${LOG_TAG}.txt"
else
  echo "No ${RESULT_MARKER} line found in ${OUT_DIR}/logcat_${LOG_TAG}.txt" >&2
  exit 4
fi
