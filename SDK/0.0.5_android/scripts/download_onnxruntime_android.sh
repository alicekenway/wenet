#!/usr/bin/env bash
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.25.1}"
SDK_ROOT="${SDK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-/tmp}"
AAR_PATH="${AAR_PATH:-${DOWNLOAD_DIR}/onnxruntime-android-${ORT_VERSION}.aar}"
ORT_MAVEN_URL="${ORT_MAVEN_URL:-https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ORT_VERSION}/onnxruntime-android-${ORT_VERSION}.aar}"

mkdir -p "${DOWNLOAD_DIR}"
curl -fL "${ORT_MAVEN_URL}" -o "${AAR_PATH}"
AAR_PATH="${AAR_PATH}" "${SDK_ROOT}/scripts/prepare_onnxruntime_android.sh"
