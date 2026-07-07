# Android SDK Modification Plan for `wenet/SDK/0.0.5`

This plan is for modifying the existing `SDK/0.0.5` C/C++ SDK so it can be cross-built for Android, packaged into a small Android test APK, and tested in the Docker Android emulator.

The first target is **not** a full car product. The first target is:

```text
Build Android x86_64 libasr_sdk.so
Build a tiny Android APK
Install it in the Docker Android emulator
Run one WAV file through the SDK
Print the ASR result to logcat
```

After that works, build `arm64-v8a` for future real car/head-unit testing.

---

## 1. Current SDK facts that affect Android

From the current `SDK/0.0.5` tree:

- The SDK is already CMake-based.
- The project builds a shared library called `asr_sdk`.
- Public headers are under `include/asr_sdk`.
- The cleanest Android boundary is the existing C API in `include/asr_sdk/c_api.h`.
- The current CMake links `dl` and `pthread` directly.
- `cmake/Options.cmake` currently has a hardcoded local WeNet root path.
- `cmake/OnnxRuntime.cmake` assumes a Linux-like ONNX Runtime layout:

```text
third_party/onnxruntime/include
third_party/onnxruntime/lib/libonnxruntime.so
```

- `cmake/WenetStaticRuntime.cmake` requires prebuilt WeNet static archives such as:

```text
runtime/onnxruntime/build/decoder/libdecoder.a
runtime/onnxruntime/build/kaldi/libkaldi-decoder.a
runtime/onnxruntime/build/frontend/libfrontend.a
...
```

For Android, every native library and static archive must be built for the same Android ABI.

Example:

```text
x86_64 emulator build:
  libasr_sdk.so          -> x86_64 Android
  libonnxruntime.so      -> x86_64 Android
  WeNet static archives  -> x86_64 Android

arm64-v8a device build:
  libasr_sdk.so          -> arm64-v8a Android
  libonnxruntime.so      -> arm64-v8a Android
  WeNet static archives  -> arm64-v8a Android
```

Do **not** link Android `libasr_sdk.so` with Linux `.a` files.

---

## 2. Target build architecture

Use two ABIs:

| ABI | Use |
|---|---|
| `x86_64` | Docker Android emulator test |
| `arm64-v8a` | Real Android phone / car computer later |

Recommended Android minimum API:

```text
android-26
```

Recommended C++ runtime:

```text
c++_shared
```

Reason: the app will likely contain multiple shared libraries:

```text
libasr_jni.so
libasr_sdk.so
libonnxruntime.so
libc++_shared.so
```

Using `c++_shared` avoids accidentally embedding separate C++ runtimes into several libraries.

---

## 3. Implementation strategy

Do the Android port in stages:

```text
Stage 1: Make CMake Android-aware
Stage 2: Support Android ONNX Runtime AAR layout
Stage 3: Cross-build WeNet static runtime for Android
Stage 4: Cross-build libasr_sdk.so
Stage 5: Create tiny JNI APK
Stage 6: Test APK in Docker Android emulator
Stage 7: Re-enable advanced decoder pieces, such as Flashlight/KenLM, only after the base path works
```

Do **not** start by porting everything at once.

First success should be a small file-based ASR test.

---

# Stage 1 — Make SDK CMake Android-aware

## 1.1 Modify `cmake/Options.cmake`

### Problem

The current options are Linux-oriented and include a hardcoded local path:

```cmake
set(ASR_SDK_WENET_ROOT "/path/to/ASR_wenet/wenet" CACHE PATH ...)
```

This will break in Docker, CI, and Android builds.

### Plan

Replace the hardcoded path with an explicit required cache variable:

```cmake
set(ASR_SDK_WENET_ROOT "" CACHE PATH "Path to the pinned alicekenway/wenet checkout")
set(ASR_SDK_WENET_BUILD_DIR "" CACHE PATH "Path to existing WeNet runtime build directory")

if(NOT ASR_SDK_WENET_ROOT)
  message(FATAL_ERROR "Please pass -DASR_SDK_WENET_ROOT=/path/to/wenet")
endif()

if(NOT ASR_SDK_WENET_BUILD_DIR)
  if(ANDROID)
    set(ASR_SDK_WENET_BUILD_DIR
        "${ASR_SDK_WENET_ROOT}/runtime/onnxruntime/out-android-${ANDROID_ABI}"
        CACHE PATH "Path to Android WeNet runtime build directory" FORCE)
  else()
    set(ASR_SDK_WENET_BUILD_DIR
        "${ASR_SDK_WENET_ROOT}/runtime/onnxruntime/out"
        CACHE PATH "Path to Linux WeNet runtime build directory" FORCE)
  endif()
endif()
```

### Android defaults

For Android, turn off CLI tools, examples, and tests by default:

```cmake
if(ANDROID)
  option(ASR_SDK_BUILD_TOOLS "Build SDK CLI tools" OFF)
  option(ASR_SDK_BUILD_EXAMPLES "Build SDK examples" OFF)
  option(ASR_SDK_BUILD_TESTS "Build SDK tests" OFF)
else()
  option(ASR_SDK_BUILD_TOOLS "Build SDK CLI tools" ON)
  option(ASR_SDK_BUILD_EXAMPLES "Build SDK examples" ON)
  option(ASR_SDK_BUILD_TESTS "Build SDK tests" ON)
endif()
```

For the first Android milestone, also default advanced decoders off:

```cmake
if(ANDROID)
  option(ASR_SDK_ENABLE_FLASHLIGHT_DECODER "Build Flashlight lexicon + KenLM decoder" OFF)
  option(ASR_SDK_ENABLE_LEGACY_WFST "Build legacy WFST comparison decoder" OFF)
else()
  option(ASR_SDK_ENABLE_FLASHLIGHT_DECODER "Build Flashlight lexicon + KenLM decoder" ON)
  option(ASR_SDK_ENABLE_LEGACY_WFST "Build legacy WFST comparison decoder" OFF)
endif()
```

Reason: Flashlight/KenLM may need extra Android work. Get the base ONNX greedy path running first.

---

## 1.2 Modify platform libraries in `CMakeLists.txt`

### Problem

The current `CMakeLists.txt` links directly with:

```cmake
dl pthread
```

This is normal on Linux, but Android should be handled through the NDK toolchain and CMake platform variables.

### Plan

Add this near the top, after project/options are loaded:

```cmake
find_package(Threads REQUIRED)

set(ASR_SDK_PLATFORM_LIBS Threads::Threads)

if(CMAKE_DL_LIBS)
  list(APPEND ASR_SDK_PLATFORM_LIBS ${CMAKE_DL_LIBS})
endif()

if(ANDROID)
  # Only needed if we add __android_log_print-based logging.
  find_library(ASR_SDK_ANDROID_LOG_LIB log)
  if(ASR_SDK_ANDROID_LOG_LIB)
    list(APPEND ASR_SDK_PLATFORM_LIBS ${ASR_SDK_ANDROID_LOG_LIB})
  endif()
endif()
```

Then replace links like:

```cmake
target_link_libraries(asr_sdk_ctc_core
  PUBLIC asr_sdk::wenet_runtime_static onnxruntime
  PRIVATE dl pthread
)
```

with:

```cmake
target_link_libraries(asr_sdk_ctc_core
  PUBLIC asr_sdk::wenet_runtime_static onnxruntime
  PRIVATE ${ASR_SDK_PLATFORM_LIBS}
)
```

And replace:

```cmake
target_link_libraries(asr_sdk
  PRIVATE asr_sdk::wenet_runtime_static asr_sdk::ctc_core onnxruntime dl pthread
)
```

with:

```cmake
target_link_libraries(asr_sdk
  PRIVATE asr_sdk::wenet_runtime_static asr_sdk::ctc_core onnxruntime ${ASR_SDK_PLATFORM_LIBS}
)
```

Also update any CLI-only executable links, but remember: CLI tools should be off for Android first.

---

## 1.3 Disable Linux RPATH logic on Android

### Problem

The current SDK sets:

```cmake
BUILD_RPATH "$ORIGIN"
INSTALL_RPATH "$ORIGIN"
```

This is useful on Linux, but not needed for normal Android APK packaging.

### Plan

Use RPATH only outside Android:

```cmake
if(NOT ANDROID)
  set_target_properties(asr_sdk PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
  )
endif()
```

For Android, the APK should package libraries under:

```text
lib/<abi>/libasr_sdk.so
lib/<abi>/libonnxruntime.so
lib/<abi>/libc++_shared.so
```

---

# Stage 2 — Support Android ONNX Runtime layout

## 2.1 Understand the ONNX Runtime Android package

ONNX Runtime for Android C/C++ is normally obtained from the Android AAR package.

After extracting the AAR, the useful files look like:

```text
onnxruntime-android.aar extracted:
  headers/
    onnxruntime_c_api.h
    onnxruntime_cxx_api.h
  jni/
    arm64-v8a/
      libonnxruntime.so
    x86_64/
      libonnxruntime.so
    armeabi-v7a/
      libonnxruntime.so
    x86/
      libonnxruntime.so
```

The current SDK expects:

```text
include/
lib/libonnxruntime.so
```

So `cmake/OnnxRuntime.cmake` must support both layouts.

---

## 2.2 Modify `cmake/OnnxRuntime.cmake`

### Desired behavior

Support this for Linux:

```text
${ASR_SDK_ONNXRUNTIME_ROOT}/include
${ASR_SDK_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
```

Support this for Android:

```text
${ASR_SDK_ONNXRUNTIME_ROOT}/headers
${ASR_SDK_ONNXRUNTIME_ROOT}/jni/${ANDROID_ABI}/libonnxruntime.so
```

Also allow a normalized internal Android layout if you prefer:

```text
${ASR_SDK_ONNXRUNTIME_ROOT}/include
${ASR_SDK_ONNXRUNTIME_ROOT}/lib/${ANDROID_ABI}/libonnxruntime.so
```

### Suggested CMake pattern

```cmake
if(NOT ASR_SDK_DYNAMIC_ONNXRUNTIME)
  message(FATAL_ERROR "ASR_SDK_DYNAMIC_ONNXRUNTIME=OFF is not supported")
endif()

if(NOT EXISTS "${ASR_SDK_ONNXRUNTIME_ROOT}")
  message(FATAL_ERROR "ONNX Runtime root not found: ${ASR_SDK_ONNXRUNTIME_ROOT}")
endif()

# Headers
if(ANDROID AND EXISTS "${ASR_SDK_ONNXRUNTIME_ROOT}/headers/onnxruntime_cxx_api.h")
  set(ASR_SDK_ONNXRUNTIME_INCLUDE_DIR "${ASR_SDK_ONNXRUNTIME_ROOT}/headers")
elseif(EXISTS "${ASR_SDK_ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  set(ASR_SDK_ONNXRUNTIME_INCLUDE_DIR "${ASR_SDK_ONNXRUNTIME_ROOT}/include")
else()
  message(FATAL_ERROR
    "ONNX Runtime headers not found. Expected either:\n"
    "  ${ASR_SDK_ONNXRUNTIME_ROOT}/headers/onnxruntime_cxx_api.h\n"
    "  ${ASR_SDK_ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
endif()

# Library
if(ANDROID)
  set(_ort_android_aar_lib
      "${ASR_SDK_ONNXRUNTIME_ROOT}/jni/${ANDROID_ABI}/libonnxruntime.so")
  set(_ort_android_normalized_lib
      "${ASR_SDK_ONNXRUNTIME_ROOT}/lib/${ANDROID_ABI}/libonnxruntime.so")

  if(EXISTS "${_ort_android_aar_lib}")
    set(ASR_SDK_ONNXRUNTIME_LIB "${_ort_android_aar_lib}")
  elseif(EXISTS "${_ort_android_normalized_lib}")
    set(ASR_SDK_ONNXRUNTIME_LIB "${_ort_android_normalized_lib}")
  else()
    message(FATAL_ERROR
      "Android ONNX Runtime library not found for ABI ${ANDROID_ABI}. Expected either:\n"
      "  ${_ort_android_aar_lib}\n"
      "  ${_ort_android_normalized_lib}")
  endif()

  set(ASR_SDK_ONNXRUNTIME_VERSION "android-aar")
else()
  set(_ort_versioned_lib "${ASR_SDK_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.${ASR_SDK_REQUIRED_ORT_VERSION}")
  set(_ort_soname_lib "${ASR_SDK_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1")
  set(_ort_plain_lib "${ASR_SDK_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")

  if(EXISTS "${_ort_versioned_lib}")
    set(ASR_SDK_ONNXRUNTIME_LIB "${_ort_versioned_lib}")
    set(ASR_SDK_ONNXRUNTIME_VERSION "${ASR_SDK_REQUIRED_ORT_VERSION}")
  elseif(EXISTS "${_ort_soname_lib}")
    set(ASR_SDK_ONNXRUNTIME_LIB "${_ort_soname_lib}")
    set(ASR_SDK_ONNXRUNTIME_VERSION "unknown")
  elseif(EXISTS "${_ort_plain_lib}")
    set(ASR_SDK_ONNXRUNTIME_LIB "${_ort_plain_lib}")
    set(ASR_SDK_ONNXRUNTIME_VERSION "unknown")
  else()
    message(FATAL_ERROR "ONNX Runtime library not found under ${ASR_SDK_ONNXRUNTIME_ROOT}/lib")
  endif()
endif()

if(NOT ANDROID
   AND NOT ASR_SDK_ONNXRUNTIME_VERSION STREQUAL ASR_SDK_REQUIRED_ORT_VERSION
   AND NOT ASR_SDK_ALLOW_ORT_VERSION_MISMATCH)
  message(FATAL_ERROR
    "ONNX Runtime version mismatch. Required ${ASR_SDK_REQUIRED_ORT_VERSION}, "
    "detected ${ASR_SDK_ONNXRUNTIME_VERSION}. "
    "Set ASR_SDK_ALLOW_ORT_VERSION_MISMATCH=ON only for local experiments.")
endif()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ASR_SDK_ONNXRUNTIME_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${ASR_SDK_ONNXRUNTIME_INCLUDE_DIR}"
)

# Linux convenience symlinks only. Do not create Android symlinks.
if(NOT ANDROID AND UNIX AND NOT APPLE)
  file(CREATE_LINK "${ASR_SDK_ONNXRUNTIME_LIB}"
       "${CMAKE_BINARY_DIR}/libonnxruntime.so.1"
       SYMBOLIC RESULT _ort_link_soname)
  file(CREATE_LINK "${ASR_SDK_ONNXRUNTIME_LIB}"
       "${CMAKE_BINARY_DIR}/libonnxruntime.so"
       SYMBOLIC RESULT _ort_link_plain)
endif()
```

---

## 2.3 Add a helper script to prepare Android ONNX Runtime

Create:

```text
SDK/0.0.5_android/scripts/prepare_onnxruntime_android.sh
```

Example:

```bash
#!/usr/bin/env bash
set -euo pipefail

ORT_VERSION="${ORT_VERSION:-1.25.1}"
SDK_ROOT="${SDK_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT_DIR="${OUT_DIR:-${SDK_ROOT}/third_party/onnxruntime-android}"

mkdir -p "${OUT_DIR}"
cd /tmp

# Option A: download manually from Maven Central, then pass AAR_PATH.
: "${AAR_PATH:?Please set AAR_PATH=/path/to/onnxruntime-android-${ORT_VERSION}.aar}"

rm -rf /tmp/onnxruntime-android-aar
mkdir -p /tmp/onnxruntime-android-aar
unzip -q "${AAR_PATH}" -d /tmp/onnxruntime-android-aar

rm -rf "${OUT_DIR}/headers" "${OUT_DIR}/jni"
cp -a /tmp/onnxruntime-android-aar/headers "${OUT_DIR}/"
cp -a /tmp/onnxruntime-android-aar/jni "${OUT_DIR}/"

find "${OUT_DIR}" -maxdepth 3 -type f | sort
```

For first implementation, manual download is acceptable. Automating Maven download can come later.

---

# Stage 3 — Cross-build WeNet static runtime for Android

## 3.1 Why this is required

The current SDK links WeNet runtime static archives. For Android, those `.a` files must be Android `.a` files.

Bad:

```text
libasr_sdk.so for Android
  links Linux libdecoder.a
```

Good:

```text
libasr_sdk.so for Android x86_64
  links Android x86_64 libdecoder.a
```

Good:

```text
libasr_sdk.so for Android arm64-v8a
  links Android arm64-v8a libdecoder.a
```

---

## 3.2 Create a WeNet runtime Android build script

Create:

```text
SDK/0.0.5_android/scripts/make_wenet_runtime_android.sh
```

Sketch:

```bash
#!/usr/bin/env bash
set -euo pipefail

ABI="${1:-x86_64}"
MINSDK="${MINSDK:-26}"

: "${ANDROID_HOME:?ANDROID_HOME is required}"
: "${ANDROID_NDK:?ANDROID_NDK is required}"
: "${WENET_ROOT:?WENET_ROOT is required}"

RUNTIME_ROOT="${WENET_ROOT}/runtime/onnxruntime"
OUTPUT_DIR="${RUNTIME_ROOT}/out-android-${ABI}"

cmake -S "${RUNTIME_ROOT}" -B "${OUTPUT_DIR}" \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ABI}" \
  -DANDROID_PLATFORM="android-${MINSDK}" \
  -DANDROID_STL=c++_shared \
  -DCMAKE_BUILD_TYPE=Release

cmake --build "${BUILD_DIR}" -j"$(nproc)"
```

This script may need adjustment depending on how `runtime/onnxruntime` currently builds its dependencies.

Acceptance check:

```bash
test -f "${WENET_ROOT}/runtime/onnxruntime/out-sdk-android-x86_64/decoder/libdecoder.a"
```

Repeat for:

```bash
./scripts/make_wenet_runtime_android.sh x86_64
./scripts/make_wenet_runtime_android.sh arm64-v8a
```

---

## 3.3 Expected dependency problems

Possible issues:

| Dependency | Possible Android issue | First response |
|---|---|---|
| OpenFST | may assume host Linux behavior | patch CMake flags or disable unused tools |
| gflags/glog | usually portable, but build paths may assume Linux | cross-build with NDK toolchain |
| wetextprocessing | may build host tools and target libs together | split host tools from target libs |
| KenLM | may link `rt` or build host tools | keep Flashlight/KenLM off first |
| pthread/dl | direct Linux link style | use platform abstraction from Stage 1 |

Important host/target distinction:

```text
Host tools:
  run on your Linux build machine
  example: model packaging tools, LM build tools

Target libraries:
  run inside Android
  example: libasr_sdk.so, libdecoder.a, libonnxruntime.so
```

Do not try to run Android-compiled tools on Linux during the build.

---

# Stage 4 — Cross-build `libasr_sdk.so` for Android

Create:

```text
SDK/0.0.5_android/scripts/make_sdk_android.sh
```

Example:

```bash
#!/usr/bin/env bash
set -euo pipefail

ABI="${1:-x86_64}"
MINSDK="${MINSDK:-26}"

: "${ANDROID_HOME:?ANDROID_HOME is required}"
: "${ANDROID_NDK:?ANDROID_NDK is required}"
: "${WENET_ROOT:?WENET_ROOT is required}"
: "${ORT_ANDROID_ROOT:?ORT_ANDROID_ROOT is required}"

SDK_ROOT="${SDK_ROOT:-${WENET_ROOT}/SDK/0.0.5}"
OUTPUT_DIR="${SDK_ROOT}/out-sdk-android-${ABI}"
WENET_BUILD_DIR="${WENET_ROOT}/runtime/onnxruntime/out-android-${ABI}"

cmake -S "${SDK_ROOT}" -B "${OUTPUT_DIR}" \
  -DCMAKE_TOOLCHAIN_FILE="${ANDROID_NDK}/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI="${ABI}" \
  -DANDROID_PLATFORM="android-${MINSDK}" \
  -DANDROID_STL=c++_shared \
  -DCMAKE_BUILD_TYPE=Release \
  -DASR_SDK_WENET_ROOT="${WENET_ROOT}" \
  -DASR_SDK_WENET_BUILD_DIR="${WENET_BUILD_DIR}" \
  -DASR_SDK_ONNXRUNTIME_ROOT="${ORT_ANDROID_ROOT}" \
  -DASR_SDK_BUILD_TOOLS=OFF \
  -DASR_SDK_BUILD_EXAMPLES=OFF \
  -DASR_SDK_BUILD_TESTS=OFF \
  -DASR_SDK_ENABLE_FLASHLIGHT_DECODER=OFF \
  -DASR_SDK_ENABLE_LEGACY_WFST=OFF

cmake --build "${BUILD_DIR}" -j"$(nproc)"

file "${BUILD_DIR}/libasr_sdk.so" || true
```

Build emulator version:

```bash
./scripts/make_sdk_android.sh x86_64
```

Build future device version:

```bash
./scripts/make_sdk_android.sh arm64-v8a
```

---

# Stage 5 — Stage Android native libraries for APK

Create a staging script:

```text
SDK/0.0.5_android/scripts/stage_android_jnilibs.sh
```

Goal output:

```text
android_demo/app/src/main/jniLibs/
  x86_64/
    libasr_sdk.so
    libonnxruntime.so
  arm64-v8a/
    libasr_sdk.so
    libonnxruntime.so
```

Script sketch:

```bash
#!/usr/bin/env bash
set -euo pipefail

ABI="${1:-x86_64}"

: "${SDK_ROOT:?SDK_ROOT is required}"
: "${ORT_ANDROID_ROOT:?ORT_ANDROID_ROOT is required}"
: "${ANDROID_DEMO_ROOT:?ANDROID_DEMO_ROOT is required}"

OUT_DIR="${ANDROID_DEMO_ROOT}/app/src/main/jniLibs/${ABI}"
mkdir -p "${OUT_DIR}"

cp "${SDK_ROOT}/out-sdk-android-${ABI}/libasr_sdk.so" "${OUT_DIR}/"

if [ -f "${ORT_ANDROID_ROOT}/jni/${ABI}/libonnxruntime.so" ]; then
  cp "${ORT_ANDROID_ROOT}/jni/${ABI}/libonnxruntime.so" "${OUT_DIR}/"
elif [ -f "${ORT_ANDROID_ROOT}/lib/${ABI}/libonnxruntime.so" ]; then
  cp "${ORT_ANDROID_ROOT}/lib/${ABI}/libonnxruntime.so" "${OUT_DIR}/"
else
  echo "Cannot find ONNX Runtime Android library for ${ABI}" >&2
  exit 1
fi

find "${OUT_DIR}" -type f -maxdepth 1 -print
```

---

# Stage 6 — Create minimal Android JNI demo

## 6.1 App goal

The app should first do only this:

```text
Start app
Copy model package from assets to app internal storage
Copy test.wav from assets to app internal storage
Call JNI function runWavTest(modelDir, wavPath)
JNI calls asr_sdk C API
Print JSON result to Android logcat
Show result on screen
```

Do not start with microphone.

---

## 6.2 Important Android file rule

Your C API takes a real filesystem path:

```c
asr_sdk_create_engine(const char* model_dir, AsrSdkEngine** out_engine)
```

Android `assets/` are not normal filesystem paths. They are packed inside the APK.

So the Android app must copy assets to internal storage first:

```text
assets/model/...  ->  context.filesDir/asr_model/...
assets/test.wav   ->  context.filesDir/test.wav
```

Then pass:

```text
/data/user/0/<package>/files/asr_model
/data/user/0/<package>/files/test.wav
```

to JNI.

---

## 6.3 Suggested Android project layout

```text
android_demo/
  settings.gradle
  build.gradle
  app/
    build.gradle
    src/main/
      AndroidManifest.xml
      java/com/example/asrdemo/
        MainActivity.kt
        AsrNative.kt
        AssetCopy.kt
      cpp/
        CMakeLists.txt
        asr_jni.cpp
      assets/
        model/
          sdk_model.json
          model.onnx
          tokens.txt
          ...
        test.wav
      jniLibs/
        x86_64/
          libasr_sdk.so
          libonnxruntime.so
        arm64-v8a/
          libasr_sdk.so
          libonnxruntime.so
```

---

## 6.4 Minimal Kotlin native wrapper

`AsrNative.kt`:

```kotlin
package com.example.asrdemo

object AsrNative {
    init {
        System.loadLibrary("asr_jni")
    }

    external fun runWavTest(modelDir: String, wavPath: String): String
}
```

---

## 6.5 Minimal JNI bridge

`asr_jni.cpp`:

```cpp
#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>

#include "asr_sdk/c_api.h"

#define LOG_TAG "ASR_TEST"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_asrdemo_AsrNative_runWavTest(
    JNIEnv* env,
    jobject,
    jstring modelDir,
    jstring wavPath) {

  const char* model_dir = env->GetStringUTFChars(modelDir, nullptr);
  const char* wav_path = env->GetStringUTFChars(wavPath, nullptr);

  LOGI("model_dir=%s", model_dir);
  LOGI("wav_path=%s", wav_path);

  // First implementation can return build info to prove the SDK loads.
  const char* build_info = asr_sdk_build_info_json();
  std::string result = build_info ? build_info : "{\"error\":\"no build info\"}";

  // Next implementation:
  // 1. create engine
  // 2. read WAV PCM16
  // 3. create stream
  // 4. accept PCM
  // 5. decode until done
  // 6. get final result JSON

  env->ReleaseStringUTFChars(modelDir, model_dir);
  env->ReleaseStringUTFChars(wavPath, wav_path);

  return env->NewStringUTF(result.c_str());
}
```

First JNI acceptance test:

```text
The app starts and logcat prints asr_sdk_build_info_json().
```

Only after that, connect WAV reading and decoding.

---

## 6.6 JNI CMakeLists

`android_demo/app/src/main/cpp/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.22.1)
project(asr_jni LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(asr_sdk SHARED IMPORTED)
set_target_properties(asr_sdk PROPERTIES
  IMPORTED_LOCATION
  "${CMAKE_CURRENT_LIST_DIR}/../jniLibs/${ANDROID_ABI}/libasr_sdk.so"
)

target_include_directories(asr_sdk INTERFACE
  "${CMAKE_CURRENT_LIST_DIR}/../../../../../SDK/0.0.5/include"
)

find_library(log-lib log)

add_library(asr_jni SHARED asr_jni.cpp)

target_link_libraries(asr_jni
  PRIVATE
    asr_sdk
    ${log-lib}
)
```

Path may need adjustment depending on where `android_demo` lives relative to `SDK/0.0.5`.

---

# Stage 7 — Docker builder environment

## 7.1 Builder image goal

The builder Docker image should contain:

```text
Ubuntu
OpenJDK
Android SDK command line tools
Android NDK
CMake
Ninja
Gradle
Git
Python
unzip
```

This Docker container builds:

```text
libasr_sdk.so
android_demo.apk
```

It does not need to run Android.

---

## 7.2 Builder run shape

Example:

```bash
docker run --rm -it \
  -v "$PWD:/work" \
  -w /work \
  asr-android-toolchain \
  bash
```

Inside the container:

```bash
export WENET_ROOT=/work/wenet
export SDK_ROOT=/work/wenet/SDK/0.0.5
export ORT_ANDROID_ROOT=/work/wenet/SDK/0.0.5/third_party/onnxruntime-android

cd "$SDK_ROOT"
./scripts/make_wenet_runtime_android.sh x86_64
./scripts/make_sdk_android.sh x86_64
```

Then stage JNI libs and build APK:

```bash
export ANDROID_DEMO_ROOT=/work/android_demo
./scripts/stage_android_jnilibs.sh x86_64
cd /work/android_demo
./gradlew assembleDebug
```

Expected output:

```text
android_demo/app/out/manual-x86_64/app-debug.apk
```

---

# Stage 8 — Test in Docker Android emulator

Assume the emulator container is already running and ADB can see it:

```bash
adb connect localhost:5555
adb devices
```

Expected:

```text
localhost:5555    device
```

Install APK:

```bash
adb install -r android_demo/app/out/manual-x86_64/app-debug.apk
```

Start app:

```bash
adb shell am start -n com.example.asrdemo/.MainActivity
```

Watch logs:

```bash
adb logcat -s ASR_TEST
```

First acceptance result:

```text
ASR_TEST: model_dir=/data/user/0/com.example.asrdemo/files/asr_model
ASR_TEST: wav_path=/data/user/0/com.example.asrdemo/files/test.wav
ASR_TEST: build info JSON printed
```

Second acceptance result:

```text
ASR_TEST: final_result={"text":"..."}
```

---

# Stage 9 — Re-enable richer decoding after base test

After the base Android APK can load `libasr_sdk.so` and decode one WAV, re-enable features one at a time.

## 9.1 Flashlight/KenLM

Do this only after the simple path works.

Potential needed changes:

- Avoid building host LM tools as Android target binaries.
- Ensure KenLM target libraries do not link Linux-only `rt` on Android.
- Cross-build only libraries needed at runtime.
- Keep LM training/package generation on Linux host.
- Package only runtime files into Android assets:

```text
lm.bin
lexicon.txt
words.txt
output_mapping.txt
sdk_model.json
```

## 9.2 Legacy WFST

Keep off unless needed:

```cmake
-DASR_SDK_ENABLE_LEGACY_WFST=OFF
```

WFST support can be revisited after Android base decoding is stable.

---

# Stage 10 — Acceptance checklist

## Build acceptance

```text
[ ] make_wenet_runtime_android.sh x86_64 creates Android x86_64 static archives
[ ] make_sdk_android.sh x86_64 creates libasr_sdk.so
[ ] make_sdk_android.sh arm64-v8a creates libasr_sdk.so
[ ] file/libreadelf confirms Android target, not Linux host
[ ] android_demo.apk contains lib/x86_64/libasr_sdk.so
[ ] android_demo.apk contains lib/x86_64/libonnxruntime.so
```

Useful checks:

```bash
readelf -h out-sdk-android-x86_64/libasr_sdk.so | grep Machine
readelf -d out-sdk-android-x86_64/libasr_sdk.so | grep NEEDED
unzip -l android_demo/app/out/manual-x86_64/app-debug.apk | grep 'lib/x86_64'
```

## Runtime acceptance

```text
[ ] adb devices shows localhost:5555 device
[ ] adb install succeeds
[ ] app starts without UnsatisfiedLinkError
[ ] app logs asr_sdk_build_info_json()
[ ] app copies model assets to filesDir
[ ] app runs test.wav
[ ] app prints final ASR JSON
```

## Not required for first milestone

```text
[ ] microphone streaming
[ ] Android Automotive UI
[ ] real car test
[ ] KenLM/Flashlight decoder
[ ] latency optimization
```

---

# Recommended implementation order

Use this exact order to avoid too many moving parts:

```text
1. Patch Options.cmake for Android defaults and no hardcoded path.
2. Patch CMakeLists.txt platform libraries: no direct dl/pthread everywhere.
3. Patch OnnxRuntime.cmake for Android AAR layout.
4. Add prepare_onnxruntime_android.sh.
5. Add make_sdk_android.sh.
6. Try SDK configure for x86_64.
7. If it fails on missing WeNet static archives, add make_wenet_runtime_android.sh.
8. Build WeNet runtime x86_64.
9. Build SDK x86_64.
10. Create android_demo with JNI that only calls asr_sdk_build_info_json().
11. Install APK in Docker emulator.
12. Connect real WAV decode.
13. Repeat build for arm64-v8a.
14. Re-enable Flashlight/KenLM if needed.
```

---

# Important risks

## Risk 1: WeNet runtime build may not be Android-ready

The SDK currently depends on WeNet static archives. If the WeNet runtime tree cannot cross-build cleanly, this becomes the main porting task.

Fallback options:

```text
Option A: Patch WeNet runtime dependencies for Android.
Option B: Add a smaller Android-only backend path that uses ONNX Runtime directly and avoids WeNet static archives.
Option C: Temporarily expose a reduced SDK mode for Android: ONNX CTC greedy only.
```

## Risk 2: ONNX Runtime version mismatch

The Linux SDK pins ONNX Runtime `1.25.1`. Android AAR version should match if possible.

For early local experiments only:

```cmake
-DASR_SDK_ALLOW_ORT_VERSION_MISMATCH=ON
```

Do not leave silent version mismatch in production.

## Risk 3: Android assets are not real paths

Always copy model assets into app internal storage before passing paths to the SDK.

## Risk 4: Emulator ABI mismatch

Docker Android emulator image is usually `x86_64`. If your APK only contains `arm64-v8a`, it will not load.

For emulator, include:

```text
lib/x86_64/libasr_sdk.so
lib/x86_64/libonnxruntime.so
```

For real car/phone, include:

```text
lib/arm64-v8a/libasr_sdk.so
lib/arm64-v8a/libonnxruntime.so
```

---

# References

- Current SDK tree: `https://github.com/alicekenway/wenet/tree/main/SDK/0.0.5`
- Current SDK `CMakeLists.txt`: `https://raw.githubusercontent.com/alicekenway/wenet/main/SDK/0.0.5/CMakeLists.txt`
- Current SDK `Options.cmake`: `https://raw.githubusercontent.com/alicekenway/wenet/main/SDK/0.0.5/cmake/Options.cmake`
- Current SDK `OnnxRuntime.cmake`: `https://raw.githubusercontent.com/alicekenway/wenet/main/SDK/0.0.5/cmake/OnnxRuntime.cmake`
- Current SDK `WenetStaticRuntime.cmake`: `https://raw.githubusercontent.com/alicekenway/wenet/main/SDK/0.0.5/cmake/WenetStaticRuntime.cmake`
- Current SDK C API: `https://raw.githubusercontent.com/alicekenway/wenet/main/SDK/0.0.5/include/asr_sdk/c_api.h`
- Android NDK CMake guide: `https://developer.android.com/ndk/guides/cmake`
- Android ABI guide: `https://developer.android.com/ndk/guides/abis`
- Android C++ runtime guide: `https://developer.android.com/ndk/guides/cpp-support`
- ONNX Runtime Android install notes: `https://onnxruntime.ai/docs/install/`
- Google Android emulator container registry notes: `https://github.com/google/android-emulator-container-scripts/blob/master/REGISTRY.MD`
