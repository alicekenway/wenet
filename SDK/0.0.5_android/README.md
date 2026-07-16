# WeNet ASR SDK 0.0.5 Android AAR Release Guide

This directory builds the Android ASR SDK as two downstream artifacts:

```text
dist/asr-sdk-android-0.0.5.aar
dist/asr-model-0.0.5.zip
```

The AAR contains the Java API, JNI wrapper, `libasr_sdk.so`, ONNX Runtime, and
the C++ runtime library. The model zip contains the runtime model package. A
downstream Android app should depend on the AAR and pass the model zip to the
SDK installer API; it should not know the internal model file layout.

## Requirements

Prepare Android build tools first:

```bash
export ROOT=/path/to/ASR_wenet
export SDK_ROOT=$ROOT/wenet/SDK/0.0.5_android
export WENET_ROOT=$ROOT/wenet

export JAVA_HOME=/tmp/asr_android_toolchain/jdk-17.0.19+10
export ANDROID_HOME=/tmp/asr_android_toolchain/android-sdk
export ANDROID_NDK=/tmp/asr_android_toolchain/android-sdk/ndk/26.3.11579264
export ORT_ANDROID_ROOT=$SDK_ROOT/third_party/onnxruntime-android
export PATH=$JAVA_HOME/bin:$ANDROID_HOME/platform-tools:$PATH
```

If ONNX Runtime Android is missing, prepare it:

```bash
SDK_ROOT=$SDK_ROOT \
  $SDK_ROOT/scripts/download_onnxruntime_android.sh
```

## Build the AAR

Build for production device and emulator ABIs:

```bash
cd $ROOT

JAVA_HOME=$JAVA_HOME \
ANDROID_HOME=$ANDROID_HOME \
ANDROID_NDK=$ANDROID_NDK \
WENET_ROOT=$WENET_ROOT \
ORT_ANDROID_ROOT=$ORT_ANDROID_ROOT \
ABIS="arm64-v8a x86_64" \
  $SDK_ROOT/scripts/make_android_aar.sh
```

Expected output:

```text
$SDK_ROOT/dist/asr-sdk-android-0.0.5.aar
```

To build one ABI only:

```bash
ABIS=x86_64 $SDK_ROOT/scripts/make_android_aar.sh
```

The AAR exposes this public Java package:

```text
com.wenet.asr.AsrSdk
com.wenet.asr.AsrRecognizer
com.wenet.asr.AsrResult
```

## Build the Model Zip

Choose a prepared 0.0.5 model package:

```bash
export MODEL_PACKAGE=$ROOT/test/0.0.5/sherpa-onnx-en-wenet-gigaspeech_int8_control_ft/package
```

The package must contain:

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

Create the zip:

```bash
MODEL_PACKAGE=$MODEL_PACKAGE \
  $SDK_ROOT/scripts/make_android_model_zip.sh
```

Expected output:

```text
$SDK_ROOT/dist/asr-model-0.0.5.zip
```

The zip stores the model files at zip root. The Android SDK extracts and
validates `sdk_model.json` during installation.

## Build the Downstream Demo APK

`android_demo` simulates a downstream app. It consumes only the generated AAR
and model zip.

```bash
JAVA_HOME=$JAVA_HOME \
ANDROID_HOME=$ANDROID_HOME \
AAR_PATH=$SDK_ROOT/dist/asr-sdk-android-0.0.5.aar \
MODEL_ZIP=$SDK_ROOT/dist/asr-model-0.0.5.zip \
  $SDK_ROOT/scripts/make_android_demo_apk.sh x86_64
```

Expected output:

```text
$SDK_ROOT/android_demo/app/out/manual-x86_64/app-debug.apk
```

## Runtime Test

Attach an emulator or device matching the APK ABI, then run:

```bash
export ADB_SERIAL=localhost:15555
export APK=$SDK_ROOT/android_demo/app/out/manual-x86_64/app-debug.apk

ADB_SERIAL=$ADB_SERIAL \
APK_PATH=$APK \
OUT_DIR=$ROOT/test/0.0.5_android/aar_consumer_demo_runtime_test \
  $SDK_ROOT/scripts/run_android_demo_test.sh
```

Expected log lines:

```text
sdk_version=0.0.5
build_info=...
final_text=...
```

## Notes

- Use `arm64-v8a` for production Android devices.
- Use `x86_64` for emulator testing.
- The model zip is large. For production apps, the downstream app can ship it
  as an asset, download it after install, or use an Android asset delivery
  mechanism. The SDK API is the same as long as the app provides the zip stream
  to `AsrSdk.installModelZip(...)`.
