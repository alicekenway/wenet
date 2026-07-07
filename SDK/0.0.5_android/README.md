# WeNet ASR SDK 0.0.5 Android Build Guide

This directory contains an Android demo for the 0.0.5 ASR SDK. It builds a
native `libasr_sdk.so`, packages it into a small Android app, copies a model
package into the APK assets, runs one WAV file, and prints the ASR result to
Android logcat.

This README is written for someone who has not used this project before.

It does not explain how to install or create an emulator. It assumes an Android
emulator or device already exists and is visible to `adb`.

## What This Android SDK Supports

The current Android build supports the 0.0.5 full package path:

- ONNX acoustic model: `model.onnx`
- Flashlight lexicon decoder
- KenLM language model: `lm.bin`
- Lexicon: `lexicon.txt`
- Acoustic tokens: `tokens.txt`
- Output words: `words.txt`
- Optional output mappings: `output_mapping.txt`, `final_output_mapping.txt`

In simple words: this APK can run the LM decoder, not only greedy CTC.

The Android build uses a reduced ONNX frontend path for portability, but still
links Flashlight-Text and KenLM into `libasr_sdk.so`.

## Directory Layout

Assume this repository is checked out at:

```bash
/path/to/ASR_wenet
```

Set these variables first:

```bash
export ROOT=/path/to/ASR_wenet
export SDK_ROOT=$ROOT/wenet/SDK/0.0.5_android
export WENET_ROOT=$ROOT/wenet
export ANDROID_DEMO_ROOT=$SDK_ROOT/android_demo
```

The final APK is built here:

```bash
$SDK_ROOT/android_demo/app/out/manual-x86_64/app-debug.apk
```

## Model Package Requirements

Prepare one model package directory before building the APK.

Example:

```bash
MODEL_PACKAGE=$ROOT/test/0.0.5/sherpa-onnx-en-wenet-gigaspeech_int8_control_ft/package
```

The package should contain:

```text
sdk_model.json
model.onnx
tokens.txt
words.txt
lexicon.txt
lm.bin
output_mapping.txt
final_output_mapping.txt
```

You mentioned acoustic model, LM, lexicon, and `token.txt`. For this SDK's full
Flashlight/KenLM path, `words.txt` is also required. The file is the decoder's
output word dictionary.

The package manifest should look like this shape:

```json
{
  "decoder_type": "flashlight_lexicon_kenlm",
  "model_path": "model.onnx",
  "tokens": "tokens.txt",
  "words": "words.txt",
  "lexicon": "lexicon.txt",
  "lm": "lm.bin",
  "mapping": "output_mapping.txt",
  "final_mapping": "final_output_mapping.txt",
  "feature_type": "kaldi",
  "blank_token": "<blank>",
  "sil_token": "▁",
  "unk_word": "<unk>",
  "sample_rate": 16000,
  "beam_size": 50,
  "beam_size_token": 15,
  "beam_threshold": 25,
  "lm_weight": 1.0,
  "word_score": -0.3,
  "unk_score": -5.0,
  "sil_score": 0.0,
  "log_add": false,
  "allow_unk": true,
  "smearing": "max",
  "nbest": 1,
  "debug": false
}
```

## Host Requirements

Install basic host tools first.

Ubuntu example:

```bash
sudo apt-get update
sudo apt-get install -y \
  ca-certificates \
  cmake \
  curl \
  file \
  make \
  tar \
  unzip \
  zip
```

The helper script below installs these Android build tools under
`/tmp/asr_android_toolchain`:

- JDK 17
- Android command-line tools
- Android SDK platform 35
- Android build-tools 35.0.0
- Android NDK r26d

Run:

```bash
cd /path/to/ASR_wenet
wenet/SDK/0.0.5_android/scripts/setup_android_toolchain.sh
```

Then export the toolchain paths:

```bash
export JAVA_HOME=/tmp/asr_android_toolchain/jdk-17.0.19+10
export ANDROID_HOME=/tmp/asr_android_toolchain/android-sdk
export ANDROID_NDK=/tmp/asr_android_toolchain/android-ndk-r26d
export PATH=$JAVA_HOME/bin:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$PATH
```

Check:

```bash
java -version
cmake --version
adb version
test -f "$ANDROID_NDK/build/cmake/android.toolchain.cmake"
```

## ONNX Runtime Android Dependency

The Android build needs ONNX Runtime Android headers and `.so` files.

Expected location:

```bash
$SDK_ROOT/third_party/onnxruntime-android
```

Check:

```bash
test -f "$SDK_ROOT/third_party/onnxruntime-android/headers/onnxruntime_c_api.h"
test -f "$SDK_ROOT/third_party/onnxruntime-android/jni/x86_64/libonnxruntime.so"
```

If missing, download and prepare it:

```bash
SDK_ROOT=$SDK_ROOT \
  $SDK_ROOT/scripts/download_onnxruntime_android.sh
```

You can also prepare a local AAR:

```bash
AAR_PATH=/path/to/onnxruntime-android-1.25.1.aar \
SDK_ROOT=$SDK_ROOT \
  $SDK_ROOT/scripts/prepare_onnxruntime_android.sh
```

## Choose the Android ABI

For the emulator used during development:

```bash
adb devices
adb -s localhost:15555 shell getprop ro.product.cpu.abi
```

Common values:

```text
x86_64
arm64-v8a
```

Use the same ABI for the SDK build and APK build.

The examples below use `x86_64`.

```bash
export ABI=x86_64
```

## Build the Native Android SDK

Build `libasr_sdk.so`:

```bash
cd /path/to/ASR_wenet

export ROOT=/path/to/ASR_wenet
export SDK_ROOT=$ROOT/wenet/SDK/0.0.5_android
export WENET_ROOT=$ROOT/wenet
export ORT_ANDROID_ROOT=$SDK_ROOT/third_party/onnxruntime-android
export ABI=x86_64

JAVA_HOME=$JAVA_HOME \
ANDROID_HOME=$ANDROID_HOME \
ANDROID_NDK=$ANDROID_NDK \
WENET_ROOT=$WENET_ROOT \
ORT_ANDROID_ROOT=$ORT_ANDROID_ROOT \
ASR_SDK_ENABLE_FLASHLIGHT_DECODER=ON \
ASR_SDK_ANDROID_REDUCED_ONNX=ON \
  $SDK_ROOT/scripts/make_sdk_android.sh $ABI
```

Expected output:

```text
[100%] Built target asr_sdk
out-sdk-android-x86_64/libasr_sdk.so: ELF 64-bit LSB shared object, x86-64
```

The library is created at:

```bash
$SDK_ROOT/out-sdk-android-x86_64/libasr_sdk.so
```

For an `arm64-v8a` device, set `ABI=arm64-v8a` and run the same command.

## Stage Native Libraries into the Demo App

Copy `libasr_sdk.so` and `libonnxruntime.so` into Android `jniLibs`:

```bash
SDK_ROOT=$SDK_ROOT \
ORT_ANDROID_ROOT=$ORT_ANDROID_ROOT \
ANDROID_DEMO_ROOT=$SDK_ROOT/android_demo \
  $SDK_ROOT/scripts/stage_android_jnilibs.sh $ABI
```

Expected files:

```bash
$SDK_ROOT/android_demo/app/src/main/jniLibs/$ABI/libasr_sdk.so
$SDK_ROOT/android_demo/app/src/main/jniLibs/$ABI/libonnxruntime.so
```

The APK build script also adds `libasr_jni.so` and `libc++_shared.so`.

## Stage Model Assets into the Demo App

Choose a test WAV file. It must be 16 kHz PCM WAV for this SDK.

Example:

```bash
export MODEL_PACKAGE=$ROOT/test/0.0.5/sherpa-onnx-en-wenet-gigaspeech_int8_control_ft/package
export TEST_WAV=$ROOT/data/ENX/test_ONEASR-2061.utf8.part/wav/000000003.wav
```

Stage package files and test WAV:

```bash
ANDROID_DEMO_ROOT=$SDK_ROOT/android_demo \
MODEL_PACKAGE=$MODEL_PACKAGE \
TEST_WAV=$TEST_WAV \
  $SDK_ROOT/scripts/stage_android_demo_assets.sh
```

Expected output includes:

```text
model/sdk_model.json
model/model.onnx
model/tokens.txt
model/words.txt
model/lexicon.txt
model/lm.bin
model/output_mapping.txt
model/final_output_mapping.txt
test.wav
```

The staged files are under:

```bash
$SDK_ROOT/android_demo/app/src/main/assets
```

## Build the APK

Build a signed debug APK:

```bash
JAVA_HOME=$JAVA_HOME \
ANDROID_HOME=$ANDROID_HOME \
ANDROID_NDK=$ANDROID_NDK \
SDK_ROOT=$SDK_ROOT \
ANDROID_DEMO_ROOT=$SDK_ROOT/android_demo \
  $SDK_ROOT/scripts/make_android_demo_apk.sh $ABI
```

Expected output:

```text
$SDK_ROOT/android_demo/app/out/manual-x86_64/app-debug.apk
```

Check the APK:

```bash
APK=$SDK_ROOT/android_demo/app/out/manual-$ABI/app-debug.apk
ls -lh "$APK"
unzip -l "$APK" | grep 'assets/model/lm.bin'
unzip -l "$APK" | grep "lib/$ABI/libasr_sdk.so"
unzip -l "$APK" | grep "lib/$ABI/libc++_shared.so"
```

Verify signing:

```bash
$ANDROID_HOME/build-tools/35.0.0/apksigner verify --verbose "$APK"
```

Expected important lines:

```text
Verifies
Verified using v2 scheme (APK Signature Scheme v2): true
Verified using v3 scheme (APK Signature Scheme v3): true
```

## Test on an Existing Emulator or Device

This README assumes the emulator already exists.

First check that ADB can see it:

```bash
adb devices
adb -s localhost:15555 shell getprop ro.build.version.release
adb -s localhost:15555 shell getprop ro.product.cpu.abi
```

Set the serial:

```bash
export ADB_SERIAL=localhost:15555
```

If your emulator uses a different serial, replace `localhost:15555` with the
value shown by `adb devices`.

Run the APK test:

```bash
ADB_SERIAL=$ADB_SERIAL \
LOGCAT_SECONDS=240 \
APK_PATH=$APK \
OUT_DIR=$ROOT/test/0.0.5_android/android_full_package_runtime_test \
  $SDK_ROOT/scripts/run_android_demo_test.sh
```

Expected output:

```text
Performing Streamed Install
Success
Starting: Intent { cmp=com.example.asrdemo/.MainActivity }
Runtime ASR log found: ...
```

Read the result:

```bash
grep 'build_info=' $ROOT/test/0.0.5_android/android_full_package_runtime_test/logcat_ASR_TEST.txt
grep 'final_text=' $ROOT/test/0.0.5_android/android_full_package_runtime_test/logcat_ASR_TEST.txt
```

For the full package path, `build_info` should show Flashlight and KenLM:

```text
"flashlight_text":{"commit":"49e163ab1e7b8108922512c294ab8513b89f404c","linkage":"static"}
"kenlm":{"commit":"5bf7b46558e1c5595bf3b8c9b0b1f9d8d257040a","linkage":"static"}
```

Example final line:

```text
final_text=AS LAW REFUSED TO ACT AS REPLACEMENT LET LOGAN A PLACE ON THE BENCH
```

## Build and Test Summary

The normal full-package flow is:

```bash
cd /path/to/ASR_wenet

export ROOT=/path/to/ASR_wenet
export SDK_ROOT=$ROOT/wenet/SDK/0.0.5_android
export WENET_ROOT=$ROOT/wenet
export ANDROID_DEMO_ROOT=$SDK_ROOT/android_demo
export ORT_ANDROID_ROOT=$SDK_ROOT/third_party/onnxruntime-android
export JAVA_HOME=/tmp/asr_android_toolchain/jdk-17.0.19+10
export ANDROID_HOME=/tmp/asr_android_toolchain/android-sdk
export ANDROID_NDK=/tmp/asr_android_toolchain/android-ndk-r26d
export PATH=$JAVA_HOME/bin:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$PATH
export ABI=x86_64
export MODEL_PACKAGE=$ROOT/test/0.0.5/sherpa-onnx-en-wenet-gigaspeech_int8_control_ft/package
export TEST_WAV=$ROOT/data/ENX/test_ONEASR-2061.utf8.part/wav/000000003.wav
export ADB_SERIAL=localhost:15555

$SDK_ROOT/scripts/setup_android_toolchain.sh

WENET_ROOT=$WENET_ROOT \
ORT_ANDROID_ROOT=$ORT_ANDROID_ROOT \
ASR_SDK_ENABLE_FLASHLIGHT_DECODER=ON \
ASR_SDK_ANDROID_REDUCED_ONNX=ON \
  $SDK_ROOT/scripts/make_sdk_android.sh $ABI

SDK_ROOT=$SDK_ROOT \
ORT_ANDROID_ROOT=$ORT_ANDROID_ROOT \
ANDROID_DEMO_ROOT=$ANDROID_DEMO_ROOT \
  $SDK_ROOT/scripts/stage_android_jnilibs.sh $ABI

ANDROID_DEMO_ROOT=$ANDROID_DEMO_ROOT \
MODEL_PACKAGE=$MODEL_PACKAGE \
TEST_WAV=$TEST_WAV \
  $SDK_ROOT/scripts/stage_android_demo_assets.sh

SDK_ROOT=$SDK_ROOT \
ANDROID_DEMO_ROOT=$ANDROID_DEMO_ROOT \
  $SDK_ROOT/scripts/make_android_demo_apk.sh $ABI

APK=$SDK_ROOT/android_demo/app/out/manual-$ABI/app-debug.apk

ADB_SERIAL=$ADB_SERIAL \
LOGCAT_SECONDS=240 \
APK_PATH=$APK \
OUT_DIR=$ROOT/test/0.0.5_android/android_full_package_runtime_test \
  $SDK_ROOT/scripts/run_android_demo_test.sh
```

## Troubleshooting

### `ANDROID_HOME is required` or `ANDROID_NDK is required`

Export the Android toolchain variables:

```bash
export JAVA_HOME=/tmp/asr_android_toolchain/jdk-17.0.19+10
export ANDROID_HOME=/tmp/asr_android_toolchain/android-sdk
export ANDROID_NDK=/tmp/asr_android_toolchain/android-ndk-r26d
export PATH=$JAVA_HOME/bin:$ANDROID_HOME/cmdline-tools/latest/bin:$ANDROID_HOME/platform-tools:$PATH
```

### `Cannot find ONNX Runtime Android library`

Check:

```bash
find $SDK_ROOT/third_party/onnxruntime-android -name libonnxruntime.so
```

If missing, run:

```bash
SDK_ROOT=$SDK_ROOT $SDK_ROOT/scripts/download_onnxruntime_android.sh
```

### `Flashlight/KenLM decoder is not enabled`

Make sure the SDK build command includes:

```bash
ASR_SDK_ENABLE_FLASHLIGHT_DECODER=ON
```

Then rebuild and restage:

```bash
$SDK_ROOT/scripts/make_sdk_android.sh $ABI
$SDK_ROOT/scripts/stage_android_jnilibs.sh $ABI
$SDK_ROOT/scripts/make_android_demo_apk.sh $ABI
```

### `No Android device/emulator is attached`

This README does not cover emulator installation. If the emulator already
exists, make sure ADB can see it:

```bash
adb devices
adb connect localhost:15555
```

Then rerun with:

```bash
ADB_SERIAL=localhost:15555 $SDK_ROOT/scripts/run_android_demo_test.sh
```

### App installs but no `final_text=` appears

The full package is large. Give it enough time:

```bash
LOGCAT_SECONDS=240 $SDK_ROOT/scripts/run_android_demo_test.sh
```

Also inspect the saved log:

```bash
cat $ROOT/test/0.0.5_android/android_full_package_runtime_test/logcat_ASR_TEST.txt
```

## Important Source Files

### Public SDK API

```text
include/asr_sdk/c_api.h
```

C ABI used by the Android JNI layer. It creates/destroys engines and streams,
accepts PCM, decodes, and returns JSON results.

```text
include/asr_sdk/asr_engine.h
include/asr_sdk/stream.h
include/asr_sdk/config.h
include/asr_sdk/result.h
```

C++ API and data types behind the C ABI.

### SDK Core

```text
src/sdk/asr_engine.cc
```

Loads `sdk_model.json`, validates the package, and chooses the decoder. For
`decoder_type=flashlight_lexicon_kenlm`, it creates the Flashlight/KenLM engine.

```text
src/sdk/c_api.cc
src/sdk/result_json.cc
```

Expose the C API and serialize ASR results to JSON.

### Model Package Loading

```text
src/package/model_package.cc
src/package/model_package_validator.cc
```

Parse `sdk_model.json`, resolve package files, validate required files, and
check decoder options.

### ONNX Acoustic Model and Features

```text
src/sherpa_onnx_wenet/ctc_onnx_backend_factory.cc
src/sherpa_onnx_wenet/wenet_ctc_onnx_backend.cc
src/sherpa_onnx_wenet/zipformer2_ctc_onnx_backend.cc
src/sherpa_onnx_wenet/whisper_feature_extractor.cc
src/sherpa_onnx_wenet/token_table.cc
```

Load the ONNX acoustic model, extract features, run CTC forward passes, and map
token IDs to token text.

### Flashlight/KenLM Decoder

```text
src/flashlight_decoder/flashlight_asr_stream.cc
src/flashlight_decoder/flashlight_ctc_stream_decoder.cc
src/flashlight_decoder/flashlight_decoder_resource.cc
src/flashlight_decoder/lexicon_loader.cc
src/flashlight_decoder/word_dictionary.cc
src/flashlight_decoder/output_sequence_mapper.cc
src/flashlight_decoder/flashlight_result_mapper.cc
```

Load `words.txt`, `lexicon.txt`, `lm.bin`, and mappings. Decode CTC log
probabilities with Flashlight's lexicon decoder and KenLM, then convert the
result to SDK output.

### Android Reduced Fallback

```text
src/android_reduced/greedy_asr_stream.cc
```

Greedy CTC fallback path. It is useful for simple package tests, but the full
0.0.5 package should use `flashlight_lexicon_kenlm`.

### Build Configuration

```text
CMakeLists.txt
cmake/Options.cmake
cmake/OnnxRuntime.cmake
cmake/WenetStaticRuntime.cmake
cmake/KenLM.cmake
cmake/FlashlightText.cmake
```

Define SDK build options, Android defaults, ONNX Runtime linkage, reduced WeNet
frontend linkage, KenLM compilation, and Flashlight-Text integration.

### Android Demo App

```text
android_demo/app/src/main/java/com/example/asrdemo/MainActivity.java
```

Copies assets to app-private storage, calls the native ASR test, and logs
`final_result=` and `final_text=`.

```text
android_demo/app/src/main/java/com/example/asrdemo/AsrSdkUsageExample.java
```

Minimal Java example for app developers. It shows how to copy the packaged
model and WAV from APK assets, call `AsrNative.runWavTest(...)`, and extract the
top-level recognized text from the returned JSON.

```text
android_demo/app/src/main/java/com/example/asrdemo/AssetCopy.java
```

Recursively copies APK assets into app-private files, because the native SDK
expects normal filesystem paths.

```text
android_demo/app/src/main/java/com/example/asrdemo/AsrNative.java
```

Loads native libraries and declares the JNI entry point.

```text
android_demo/app/src/main/cpp/asr_jni.cpp
```

JNI bridge. It reads the test WAV, creates the SDK engine, feeds PCM, runs
decode, and returns the final JSON result.

```text
android_demo/app/src/main/cpp/CMakeLists.txt
```

Builds `libasr_jni.so` and links it against the staged `libasr_sdk.so`.

### Helper Scripts

```text
scripts/setup_android_toolchain.sh
```

Installs the Android build toolchain under `/tmp/asr_android_toolchain`.

```text
scripts/download_onnxruntime_android.sh
scripts/prepare_onnxruntime_android.sh
```

Download or unpack ONNX Runtime Android into `third_party/onnxruntime-android`.

```text
scripts/make_sdk_android.sh
```

Builds `libasr_sdk.so` for one ABI.

```text
scripts/stage_android_jnilibs.sh
```

Copies `libasr_sdk.so` and `libonnxruntime.so` into the demo app's `jniLibs`.

```text
scripts/stage_android_demo_assets.sh
```

Copies the model package and test WAV into the demo app's APK assets.

```text
scripts/make_android_demo_apk.sh
```

Builds `libasr_jni.so`, packages assets and native libraries, aligns, signs,
and verifies the debug APK.

```text
scripts/run_android_demo_test.sh
```

Installs the APK on an existing emulator/device, launches the app, captures
`ASR_TEST` logcat output, and waits for `final_result=`.
