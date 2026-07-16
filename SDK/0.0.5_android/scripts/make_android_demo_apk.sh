#!/usr/bin/env bash
set -euo pipefail

ABI="${1:-x86_64}"
MINSDK="${MINSDK:-26}"
TARGETSDK="${TARGETSDK:-35}"
VERSION="${VERSION:-0.0.5}"

: "${ANDROID_HOME:?ANDROID_HOME is required}"

SDK_ROOT="${SDK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="${REPO_ROOT:-$(cd "${SDK_ROOT}/../../.." && pwd)}"
ANDROID_DEMO_ROOT="${ANDROID_DEMO_ROOT:-${SDK_ROOT}/android_demo}"
APP_ROOT="${ANDROID_DEMO_ROOT}/app"
OUTPUT_ROOT="${OUTPUT_ROOT:-${APP_ROOT}/out}"
WORK_DIR="${OUTPUT_ROOT}/manual-${ABI}"
AAR_PATH="${AAR_PATH:-${SDK_ROOT}/dist/asr-sdk-android-${VERSION}.aar}"
MODEL_ZIP="${MODEL_ZIP:-${SDK_ROOT}/dist/asr-model-${VERSION}.zip}"
TEST_WAV="${TEST_WAV:-${REPO_ROOT}/data/ENX/test_ONEASR-2061.utf8.part/wav/000000003.wav}"

if [ -n "${JAVA_HOME:-}" ]; then
  export PATH="${JAVA_HOME}/bin:${PATH}"
fi

if [ ! -f "${AAR_PATH}" ]; then
  echo "Missing AAR: ${AAR_PATH}" >&2
  echo "Run scripts/make_android_aar.sh first." >&2
  exit 1
fi
if [ ! -f "${MODEL_ZIP}" ]; then
  echo "Missing model zip: ${MODEL_ZIP}" >&2
  echo "Run scripts/make_android_model_zip.sh first." >&2
  exit 1
fi
if [ ! -f "${TEST_WAV}" ]; then
  echo "Missing test WAV: ${TEST_WAV}" >&2
  exit 1
fi

ANDROID_JAR="${ANDROID_HOME}/platforms/android-${TARGETSDK}/android.jar"
if [ ! -f "${ANDROID_JAR}" ]; then
  ANDROID_JAR="$(find "${ANDROID_HOME}/platforms" -maxdepth 2 -name android.jar | sort -V | tail -n 1)"
fi
: "${ANDROID_JAR:?No android.jar found under ANDROID_HOME/platforms}"

BUILD_TOOLS_DIR="$(find "${ANDROID_HOME}/build-tools" -mindepth 1 -maxdepth 1 -type d | sort -V | tail -n 1)"
: "${BUILD_TOOLS_DIR:?No Android build-tools directory found}"

AAPT2="${BUILD_TOOLS_DIR}/aapt2"
D8="${BUILD_TOOLS_DIR}/d8"
ZIPALIGN="${BUILD_TOOLS_DIR}/zipalign"
APKSIGNER="${BUILD_TOOLS_DIR}/apksigner"

rm -rf "${WORK_DIR}"
mkdir -p "${WORK_DIR}"/{aar,assets,gen,classes,dex,apk/lib/${ABI}}

unzip -q "${AAR_PATH}" -d "${WORK_DIR}/aar"
test -f "${WORK_DIR}/aar/classes.jar"
test -f "${WORK_DIR}/aar/jni/${ABI}/libasr_sdk.so"
test -f "${WORK_DIR}/aar/jni/${ABI}/libasr_jni.so"
test -f "${WORK_DIR}/aar/jni/${ABI}/libonnxruntime.so"

cp "${MODEL_ZIP}" "${WORK_DIR}/assets/asr-model.zip"
cp "${TEST_WAV}" "${WORK_DIR}/assets/test.wav"
cp "${WORK_DIR}/aar/jni/${ABI}/"*.so "${WORK_DIR}/apk/lib/${ABI}/"

"${AAPT2}" link \
  -I "${ANDROID_JAR}" \
  --manifest "${APP_ROOT}/src/main/AndroidManifest.xml" \
  --java "${WORK_DIR}/gen" \
  --min-sdk-version "${MINSDK}" \
  --target-sdk-version "${TARGETSDK}" \
  -A "${WORK_DIR}/assets" \
  -o "${WORK_DIR}/base.apk"

javac -encoding UTF-8 \
  -source 1.8 -target 1.8 \
  -bootclasspath "${ANDROID_JAR}" \
  -classpath "${WORK_DIR}/aar/classes.jar" \
  -d "${WORK_DIR}/classes" \
  $(find "${WORK_DIR}/gen" "${APP_ROOT}/src/main/java" -name '*.java' | sort)

"${D8}" --min-api "${MINSDK}" --output "${WORK_DIR}/dex" \
  "${WORK_DIR}/aar/classes.jar" \
  $(find "${WORK_DIR}/classes" -name '*.class' | sort)

cp "${WORK_DIR}/base.apk" "${WORK_DIR}/unsigned.apk"
(
  cd "${WORK_DIR}/dex"
  zip -q "${WORK_DIR}/unsigned.apk" classes.dex
)
(
  cd "${WORK_DIR}/apk"
  zip -qr "${WORK_DIR}/unsigned.apk" lib
)

KEYSTORE="${ANDROID_DEMO_ROOT}/debug.keystore"
if [ ! -f "${KEYSTORE}" ]; then
  keytool -genkeypair -v \
    -keystore "${KEYSTORE}" \
    -storepass android \
    -alias androiddebugkey \
    -keypass android \
    -keyalg RSA \
    -keysize 2048 \
    -validity 10000 \
    -dname "CN=Android Debug,O=Android,C=US"
fi

"${ZIPALIGN}" -f -p 4 "${WORK_DIR}/unsigned.apk" "${WORK_DIR}/aligned.apk"
"${APKSIGNER}" sign \
  --ks "${KEYSTORE}" \
  --ks-pass pass:android \
  --key-pass pass:android \
  --out "${WORK_DIR}/app-debug.apk" \
  "${WORK_DIR}/aligned.apk"

"${APKSIGNER}" verify "${WORK_DIR}/app-debug.apk"
echo "${WORK_DIR}/app-debug.apk"
