#!/usr/bin/env bash
set -euo pipefail

TOOLCHAIN_ROOT="${TOOLCHAIN_ROOT:-/tmp/asr_android_toolchain}"
ANDROID_CMDLINE_TOOLS_URL="${ANDROID_CMDLINE_TOOLS_URL:-https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip}"
JDK_URL="${JDK_URL:-https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.19%2B10/OpenJDK17U-jdk_x64_linux_hotspot_17.0.19_10.tar.gz}"
ANDROID_PLATFORM="${ANDROID_PLATFORM:-android-35}"
ANDROID_BUILD_TOOLS="${ANDROID_BUILD_TOOLS:-35.0.0}"
ANDROID_NDK_VERSION="${ANDROID_NDK_VERSION:-26.3.11579264}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
usage: setup_android_toolchain.sh

Downloads or reuses an Android command-line toolchain under TOOLCHAIN_ROOT.

Common overrides:
  TOOLCHAIN_ROOT=/tmp/asr_android_toolchain
  ANDROID_PLATFORM=android-35
  ANDROID_BUILD_TOOLS=35.0.0
  ANDROID_NDK_VERSION=26.3.11579264
EOF
  exit 0
fi

ANDROID_HOME="${ANDROID_HOME:-${TOOLCHAIN_ROOT}/android-sdk}"
JAVA_HOME="${JAVA_HOME:-${TOOLCHAIN_ROOT}/jdk-17.0.19+10}"
ANDROID_NDK="${ANDROID_NDK:-${TOOLCHAIN_ROOT}/android-ndk-r26d}"
DOWNLOAD_DIR="${DOWNLOAD_DIR:-${TOOLCHAIN_ROOT}/downloads}"

mkdir -p "${TOOLCHAIN_ROOT}" "${DOWNLOAD_DIR}" "${ANDROID_HOME}/cmdline-tools"

if [ ! -x "${JAVA_HOME}/bin/java" ]; then
  JDK_ARCHIVE="${DOWNLOAD_DIR}/jdk-17.0.19+10.tar.gz"
  curl -fL "${JDK_URL}" -o "${JDK_ARCHIVE}"
  rm -rf "${JAVA_HOME}"
  tar -xzf "${JDK_ARCHIVE}" -C "${TOOLCHAIN_ROOT}"
fi

CMDLINE_TOOLS="${ANDROID_HOME}/cmdline-tools/latest"
if [ ! -x "${CMDLINE_TOOLS}/bin/sdkmanager" ]; then
  CMDLINE_TOOLS_ZIP="${DOWNLOAD_DIR}/android-commandline-tools.zip"
  curl -fL "${ANDROID_CMDLINE_TOOLS_URL}" -o "${CMDLINE_TOOLS_ZIP}"
  rm -rf "${TOOLCHAIN_ROOT}/android-commandline-tools-unpack" \
         "${CMDLINE_TOOLS}"
  mkdir -p "${TOOLCHAIN_ROOT}/android-commandline-tools-unpack"
  unzip -q "${CMDLINE_TOOLS_ZIP}" \
    -d "${TOOLCHAIN_ROOT}/android-commandline-tools-unpack"
  mv "${TOOLCHAIN_ROOT}/android-commandline-tools-unpack/cmdline-tools" \
     "${CMDLINE_TOOLS}"
fi

export JAVA_HOME
export ANDROID_HOME
export ANDROID_SDK_ROOT="${ANDROID_HOME}"
export PATH="${JAVA_HOME}/bin:${CMDLINE_TOOLS}/bin:${ANDROID_HOME}/platform-tools:${PATH}"

set +o pipefail
yes | sdkmanager --sdk_root="${ANDROID_HOME}" --licenses >/dev/null
set -o pipefail
sdkmanager --sdk_root="${ANDROID_HOME}" \
  "platform-tools" \
  "platforms;${ANDROID_PLATFORM}" \
  "build-tools;${ANDROID_BUILD_TOOLS}" \
  "ndk;${ANDROID_NDK_VERSION}"

if [ ! -d "${ANDROID_NDK}" ]; then
  INSTALLED_NDK="${ANDROID_HOME}/ndk/${ANDROID_NDK_VERSION}"
  if [ -d "${INSTALLED_NDK}" ]; then
    ln -sfn "${INSTALLED_NDK}" "${ANDROID_NDK}"
  fi
fi

cat <<'EOF'
# Add these to your shell before building.
# If you used a custom TOOLCHAIN_ROOT for this script, set the same value here.
export TOOLCHAIN_ROOT=${TOOLCHAIN_ROOT:-/tmp/asr_android_toolchain}
export JAVA_HOME=${TOOLCHAIN_ROOT}/jdk-17.0.19+10
export ANDROID_HOME=${TOOLCHAIN_ROOT}/android-sdk
export ANDROID_NDK=${TOOLCHAIN_ROOT}/android-ndk-r26d
export PATH=${JAVA_HOME}/bin:${ANDROID_HOME}/cmdline-tools/latest/bin:${ANDROID_HOME}/platform-tools:${PATH}
EOF
