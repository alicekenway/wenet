#!/usr/bin/env bash
set -euo pipefail

ABI="${1:-x86_64}"
MINSDK="${MINSDK:-26}"
TARGETSDK="${TARGETSDK:-35}"

: "${ANDROID_HOME:?ANDROID_HOME is required}"
: "${ANDROID_NDK:?ANDROID_NDK is required}"

SDK_ROOT="${SDK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ANDROID_DEMO_ROOT="${ANDROID_DEMO_ROOT:-${SDK_ROOT}/android_demo}"
APP_ROOT="${ANDROID_DEMO_ROOT}/app"
OUTPUT_ROOT="${OUTPUT_ROOT:-${APP_ROOT}/out}"
WORK_DIR="${OUTPUT_ROOT}/manual-${ABI}"

if [ -n "${JAVA_HOME:-}" ]; then
  export PATH="${JAVA_HOME}/bin:${PATH}"
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
mkdir -p "${WORK_DIR}"/{native,gen,classes,dex,apk/lib/${ABI}}

cmake -S "${APP_ROOT}/src/main/cpp" -B "${WORK_DIR}/native" \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ABI}" \
  -DANDROID_PLATFORM="android-${MINSDK}" \
  -DANDROID_STL=c++_shared \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "${WORK_DIR}/native" -j"$(nproc)"

cp "${WORK_DIR}/native/libasr_jni.so" "${WORK_DIR}/apk/lib/${ABI}/"
cp "${APP_ROOT}/src/main/jniLibs/${ABI}/libasr_sdk.so" "${WORK_DIR}/apk/lib/${ABI}/"
cp "${APP_ROOT}/src/main/jniLibs/${ABI}/libonnxruntime.so" "${WORK_DIR}/apk/lib/${ABI}/"
case "${ABI}" in
  x86_64)
    NDK_TRIPLE="x86_64-linux-android"
    ;;
  arm64-v8a)
    NDK_TRIPLE="aarch64-linux-android"
    ;;
  x86)
    NDK_TRIPLE="i686-linux-android"
    ;;
  armeabi-v7a)
    NDK_TRIPLE="arm-linux-androideabi"
    ;;
  *)
    echo "Unsupported ABI for libc++ lookup: ${ABI}" >&2
    exit 1
    ;;
esac
CXX_SHARED="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/${NDK_TRIPLE}/${MINSDK}/libc++_shared.so"
if [ ! -f "${CXX_SHARED}" ]; then
  CXX_SHARED="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/${NDK_TRIPLE}/libc++_shared.so"
fi
if [ ! -f "${CXX_SHARED}" ]; then
  CXX_SHARED="$(find -L "${ANDROID_NDK}" -path "*/${NDK_TRIPLE}*/libc++_shared.so" | sort -V | tail -n 1)"
fi
if [ -n "${CXX_SHARED}" ] && [ -f "${CXX_SHARED}" ]; then
  cp "${CXX_SHARED}" "${WORK_DIR}/apk/lib/${ABI}/"
else
  echo "Cannot find libc++_shared.so for ${ABI}" >&2
  exit 1
fi

"${AAPT2}" link \
  -I "${ANDROID_JAR}" \
  --manifest "${APP_ROOT}/src/main/AndroidManifest.xml" \
  --java "${WORK_DIR}/gen" \
  --min-sdk-version "${MINSDK}" \
  --target-sdk-version "${TARGETSDK}" \
  -A "${APP_ROOT}/src/main/assets" \
  -o "${WORK_DIR}/base.apk"

javac -encoding UTF-8 \
  -source 1.8 -target 1.8 \
  -bootclasspath "${ANDROID_JAR}" \
  -d "${WORK_DIR}/classes" \
  $(find "${WORK_DIR}/gen" "${APP_ROOT}/src/main/java" -name '*.java' | sort)

"${D8}" --min-api "${MINSDK}" --output "${WORK_DIR}/dex" \
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
