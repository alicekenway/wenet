#!/usr/bin/env bash
set -euo pipefail

VERSION="${VERSION:-0.0.5}"
MINSDK="${MINSDK:-26}"
TARGETSDK="${TARGETSDK:-35}"
ABIS="${ABIS:-arm64-v8a x86_64}"
if [ "$#" -gt 0 ]; then
  ABIS="$*"
fi

: "${ANDROID_HOME:?ANDROID_HOME is required}"
: "${ANDROID_NDK:?ANDROID_NDK is required}"
: "${WENET_ROOT:?WENET_ROOT is required}"
: "${ORT_ANDROID_ROOT:?ORT_ANDROID_ROOT is required}"

SDK_ROOT="${SDK_ROOT:-${WENET_ROOT}/SDK/0.0.5_android}"
AAR_ROOT="${AAR_ROOT:-${SDK_ROOT}/android_aar}"
BUILD_ROOT="${BUILD_ROOT:-${SDK_ROOT}/build-aar}"
DIST_DIR="${DIST_DIR:-${SDK_ROOT}/dist}"
OUTPUT_AAR="${OUTPUT_AAR:-${DIST_DIR}/asr-sdk-android-${VERSION}.aar}"
ASR_SDK_SKIP_NATIVE_BUILD="${ASR_SDK_SKIP_NATIVE_BUILD:-OFF}"

if [ -n "${JAVA_HOME:-}" ]; then
  export PATH="${JAVA_HOME}/bin:${PATH}"
fi

ANDROID_JAR="${ANDROID_HOME}/platforms/android-${TARGETSDK}/android.jar"
if [ ! -f "${ANDROID_JAR}" ]; then
  ANDROID_JAR="$(find "${ANDROID_HOME}/platforms" -maxdepth 2 -name android.jar | sort -V | tail -n 1)"
fi
: "${ANDROID_JAR:?No android.jar found under ANDROID_HOME/platforms}"

find_onnxruntime_so() {
  local abi="$1"
  if [ -f "${ORT_ANDROID_ROOT}/jni/${abi}/libonnxruntime.so" ]; then
    printf '%s\n' "${ORT_ANDROID_ROOT}/jni/${abi}/libonnxruntime.so"
  elif [ -f "${ORT_ANDROID_ROOT}/lib/${abi}/libonnxruntime.so" ]; then
    printf '%s\n' "${ORT_ANDROID_ROOT}/lib/${abi}/libonnxruntime.so"
  else
    echo "Cannot find ONNX Runtime Android library for ${abi}" >&2
    return 1
  fi
}

find_libcxx_shared() {
  local abi="$1"
  local triple
  case "${abi}" in
    x86_64)
      triple="x86_64-linux-android"
      ;;
    arm64-v8a)
      triple="aarch64-linux-android"
      ;;
    x86)
      triple="i686-linux-android"
      ;;
    armeabi-v7a)
      triple="arm-linux-androideabi"
      ;;
    *)
      echo "Unsupported ABI for libc++ lookup: ${abi}" >&2
      return 1
      ;;
  esac

  local cxx_shared="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/${triple}/${MINSDK}/libc++_shared.so"
  if [ ! -f "${cxx_shared}" ]; then
    cxx_shared="${ANDROID_NDK}/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/${triple}/libc++_shared.so"
  fi
  if [ ! -f "${cxx_shared}" ]; then
    cxx_shared="$(find -L "${ANDROID_NDK}" -path "*/${triple}*/libc++_shared.so" | sort -V | tail -n 1)"
  fi
  if [ -z "${cxx_shared}" ] || [ ! -f "${cxx_shared}" ]; then
    echo "Cannot find libc++_shared.so for ${abi}" >&2
    return 1
  fi
  printf '%s\n' "${cxx_shared}"
}

stage_native_sdk_for_abi() {
  local abi="$1"
  local sdk_output_dir="${SDK_ROOT}/out-sdk-android-${abi}"
  if [ ! -f "${sdk_output_dir}/libasr_sdk.so" ]; then
    if [ "${ASR_SDK_SKIP_NATIVE_BUILD}" = "ON" ]; then
      echo "Missing ${sdk_output_dir}/libasr_sdk.so and ASR_SDK_SKIP_NATIVE_BUILD=ON" >&2
      return 1
    fi
    SDK_ROOT="${SDK_ROOT}" \
    WENET_ROOT="${WENET_ROOT}" \
    ORT_ANDROID_ROOT="${ORT_ANDROID_ROOT}" \
      "${SDK_ROOT}/scripts/make_sdk_android.sh" "${abi}"
  fi

  local out_dir="${BUILD_ROOT}/staged_jni/${abi}"
  mkdir -p "${out_dir}"
  cp "${sdk_output_dir}/libasr_sdk.so" "${out_dir}/"
  cp "$(find_onnxruntime_so "${abi}")" "${out_dir}/"
  cp "$(find_libcxx_shared "${abi}")" "${out_dir}/"
}

build_jni_for_abi() {
  local abi="$1"
  local build_dir="${BUILD_ROOT}/native-${abi}"
  cmake -S "${AAR_ROOT}/src/main/cpp" -B "${build_dir}" \
    -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="${abi}" \
    -DANDROID_PLATFORM="android-${MINSDK}" \
    -DANDROID_STL=c++_shared \
    -DCMAKE_BUILD_TYPE=Release \
    -DASR_JNI_SDK_ROOT="${SDK_ROOT}" \
    -DASR_JNI_LIB_DIR="${BUILD_ROOT}/staged_jni"
  cmake --build "${build_dir}" -j"$(nproc)"
  cp "${build_dir}/libasr_jni.so" "${BUILD_ROOT}/staged_jni/${abi}/"
}

compile_java_api() {
  mkdir -p "${BUILD_ROOT}/classes"
  javac -encoding UTF-8 \
    -source 1.8 -target 1.8 \
    -bootclasspath "${ANDROID_JAR}" \
    -d "${BUILD_ROOT}/classes" \
    $(find "${AAR_ROOT}/src/main/java" -name '*.java' | sort)
  jar cf "${BUILD_ROOT}/classes.jar" -C "${BUILD_ROOT}/classes" .
}

pack_aar() {
  local aar_dir="${BUILD_ROOT}/aar"
  rm -rf "${aar_dir}"
  mkdir -p "${aar_dir}/jni"

  cp "${AAR_ROOT}/src/main/AndroidManifest.xml" "${aar_dir}/AndroidManifest.xml"
  cp "${BUILD_ROOT}/classes.jar" "${aar_dir}/classes.jar"
  cp "${AAR_ROOT}/consumer-rules.pro" "${aar_dir}/proguard.txt"
  : > "${aar_dir}/R.txt"

  for abi in ${ABIS}; do
    mkdir -p "${aar_dir}/jni/${abi}"
    cp "${BUILD_ROOT}/staged_jni/${abi}/libasr_sdk.so" "${aar_dir}/jni/${abi}/"
    cp "${BUILD_ROOT}/staged_jni/${abi}/libonnxruntime.so" "${aar_dir}/jni/${abi}/"
    cp "${BUILD_ROOT}/staged_jni/${abi}/libasr_jni.so" "${aar_dir}/jni/${abi}/"
    cp "${BUILD_ROOT}/staged_jni/${abi}/libc++_shared.so" "${aar_dir}/jni/${abi}/"
  done

  mkdir -p "${DIST_DIR}"
  rm -f "${OUTPUT_AAR}"
  (
    cd "${aar_dir}"
    zip -qr "${OUTPUT_AAR}" .
  )
  unzip -l "${OUTPUT_AAR}" | grep -E 'classes.jar|libasr_sdk.so|libasr_jni.so|libonnxruntime.so' >/dev/null
  echo "${OUTPUT_AAR}"
}

rm -rf "${BUILD_ROOT}"
mkdir -p "${BUILD_ROOT}"

for abi in ${ABIS}; do
  stage_native_sdk_for_abi "${abi}"
  build_jni_for_abi "${abi}"
done

compile_java_api
pack_aar
