# WeNet Static Runtime + Dynamic ONNX Runtime ASR SDK Engineering Plan

> **Purpose for Codex:** Build a production-shaped C++ streaming ASR SDK that wraps **WeNet's ONNX runtime path**. WeNet runtime code is vendored and linked statically into our SDK. ONNX Runtime is linked dynamically and bundled with the release package.
>
> This plan supersedes the earlier wrapper plan where WeNet was simply treated as a runtime dependency. The final dependency strategy is:
>
> ```text
> WeNet: pinned full source submodule, build only required runtime targets, link statically into our SDK.
> ONNX Runtime: shared library dependency, bundled with our release, not exposed through public SDK headers.
> SDK users: include/link only our SDK API; they should not include WeNet or ONNX Runtime headers.
> ```

---

## 1. Goal

Create a C++ streaming ASR SDK for embedded and industrial integration:

```text
Application / embedded product
  -> our public C++ SDK API or C ABI
  -> our SDK engine and stream/session wrapper
  -> our private WeNet bridge layer
  -> statically linked WeNet runtime code
  -> dynamically linked ONNX Runtime
  -> WeNet CTC prefix or CTC WFST decoder
  -> normalized partial/final ASR result
```

The SDK should support:

```text
16 kHz PCM streaming audio
WeNet ONNX model package
CTC output
optional n-gram WFST / TLG.fst decoding
partial result
final result
continuous decoding
C++ API
C ABI
CLI demos
tests
benchmarking
embedded integration examples
```

The SDK should **not** reimplement these in the MVP:

```text
fbank feature extraction
ONNX model loading logic
encoder cache handling
CTC log-prob generation
CTC prefix beam search
CTC WFST beam search
FST decoder internals
WeNet post-processing internals
```

The SDK should provide:

```text
clean product API
stable C ABI
model package validation
config translation
stream lifecycle management
error handling
result normalization
release packaging
symbol hiding
embedded-friendly examples
```

---

## 2. Final dependency decision

### 2.1 Production default

```text
Production dependency strategy:

our_asr_sdk
  ├── public API: our C++ and C ABI
  ├── private adapter: our WeNet bridge layer
  ├── static internal dependency: selected WeNet runtime targets
  ├── dynamic runtime dependency: libonnxruntime.so / onnxruntime.dll / libonnxruntime.dylib
  └── model package: ONNX files, units, optional words/TLG.fst, SDK manifest
```

### 2.2 Why this design

| Requirement | Design choice |
|---|---|
| Reproducible ASR behavior | Pin WeNet to an exact commit and link its runtime code statically. |
| Maintainable upstream upgrade | Keep the full WeNet repository as a submodule instead of copying selected files. |
| Reasonable binary size | Build only required WeNet runtime targets; do not build training, servers, graph tools, tests, or CLIs for release. |
| Practical inference dependency | Use ONNX Runtime as a dynamic library. Fully static ONNX Runtime is optional later. |
| Clean SDK boundary | Public headers expose only our SDK types. WeNet, OpenFst, and ONNX Runtime stay private. |
| Embedded deployment | Release package contains one SDK library, one ONNX Runtime shared library, model files, and licenses. |

### 2.3 Important distinction

Vendoring the full WeNet source tree is **not** the same as shipping or linking the full WeNet project.

```text
Good:
  third_party/wenet/ contains the full pinned source tree.
  CMake builds only the minimal ONNX runtime path.
  Release package does not include unnecessary WeNet source or tools.

Bad:
  Manually copy a small random set of WeNet .cc/.h files into our repo.
  Maintain a private mini-WeNet fork from day one.
```

Manual source extraction may look clean at first, but it becomes hard to maintain because WeNet runtime files depend on decoder, frontend, post-processing, Kaldi-compatible decoder utilities, OpenFst-related code, logging, and build flags. The safer approach is **full submodule + minimal build target + adapter boundary**.

---

## 3. Architecture

```text
+--------------------------------+
| Product application             |
| C++ / C / Rust / JNI / etc.     |
+----------------+---------------+
                 |
                 v
+--------------------------------+
| Public SDK API                  |
| AsrEngine / AsrStream / C ABI   |
+----------------+---------------+
                 |
                 v
+--------------------------------+
| SDK implementation              |
| config, package, status, logs   |
+----------------+---------------+
                 |
                 v
+--------------------------------+
| Private WeNet bridge            |
| WenetRuntimeBridge              |
| WenetStreamAdapter              |
| WenetConfigMapper               |
| WenetResultMapper               |
+----------------+---------------+
                 |
                 v
+--------------------------------+
| Static WeNet runtime code       |
| FeaturePipeline                 |
| OnnxAsrModel                    |
| AsrDecoder                      |
| CTC prefix / CTC WFST decoder   |
| post processor                  |
+----------------+---------------+
                 |
                 v
+--------------------------------+
| Dynamic ONNX Runtime            |
| libonnxruntime.so / dll / dylib |
+--------------------------------+
```

### Ownership model

```text
AsrEngine
  owns shared immutable resources:
    - SDK config
    - model package paths
    - mapped WeNet feature config
    - mapped WeNet decode options
    - WeNet DecodeResource or equivalent shared resource object
    - model/package metadata
    - SDK-level logger and metrics config

AsrStream
  owns mutable per-stream resources:
    - WeNet FeaturePipeline or equivalent stream frontend
    - WeNet AsrDecoder or equivalent decoder object
    - partial result cache
    - final result cache
    - input-finished flag
    - segment index
    - stream-local error state
    - optional per-stream metric counters
```

Reasoning:

```text
- Model, FST graph, units, words, and config are expensive and should be loaded once per engine.
- Each stream needs separate decoding state and feature/audio queue.
- The public SDK should make stream state explicit and safe to reset.
```

---

## 4. Repository layout

```text
wenet-static-ort-dynamic-sdk/
  CMakeLists.txt
  README.md
  LICENSE
  NOTICE

  cmake/
    Options.cmake
    OnnxRuntimeDynamic.cmake
    WenetStaticRuntime.cmake
    SymbolVisibility.cmake
    InstallRules.cmake
    Sanitizers.cmake
    Toolchain-aarch64-linux.cmake

  third_party/
    wenet/                         # full pinned git submodule
    onnxruntime/                   # optional local ORT package: include/ + lib/
    README.md

  patches/
    wenet/
      README.md
      0001-minimal-static-runtime-target.patch        # only if required
      0002-avoid-building-unused-tools.patch          # only if required

  include/
    asr_sdk/
      asr_engine.h
      asr_stream.h
      asr_result.h
      asr_config.h
      asr_status.h
      c_api.h
      export.h
      version.h

  src/
    sdk/
      asr_engine.cc
      asr_stream.cc
      asr_result.cc
      asr_config.cc
      asr_status.cc
      c_api.cc
      version.cc

    wenet_bridge/
      wenet_runtime_bridge.h
      wenet_runtime_bridge.cc
      wenet_stream_adapter.h
      wenet_stream_adapter.cc
      wenet_config_mapper.h
      wenet_config_mapper.cc
      wenet_result_mapper.h
      wenet_result_mapper.cc
      wenet_error_boundary.h
      wenet_error_boundary.cc

    package/
      model_package.h
      model_package.cc
      model_package_validator.h
      model_package_validator.cc
      checksum.h
      checksum.cc

    audio/
      wav_reader.h
      wav_reader.cc
      resampler.h
      resampler.cc
      ring_buffer.h
      ring_buffer.cc

    utils/
      json.h
      file_utils.h
      file_utils.cc
      logging.h
      logging.cc
      timer.h
      timer.cc
      metrics.h
      metrics.cc

  tools/
    asr_stream_file.cc
    asr_stream_mic.cc
    benchmark.cc
    inspect_package.cc
    compare_with_wenet.cc
    print_build_info.cc

  examples/
    cpp/
      minimal_streaming.cc
      continuous_decoding.cc
    c/
      simple_c_api.c
    embedded/
      audio_callback_pattern.cc

  scripts/
    add_wenet_submodule.sh
    pin_wenet_commit.sh
    fetch_onnxruntime.sh
    build_wenet_runtime_sanity.sh
    export_wenet_onnx.sh
    build_tlg_with_wenet.sh
    package_model.py
    validate_package.py
    package_release.sh
    verify_release_deps.sh

  tests/
    unit/
      test_config_mapper.cc
      test_model_package_validator.cc
      test_result_mapper.cc
      test_c_api_lifetime.cc
      test_status.cc
    integration/
      test_wenet_bridge_load.cc
      test_decode_wav_prefix.cc
      test_decode_wav_wfst.cc
      test_compare_with_wenet_decoder_main.cc
      test_dynamic_ort_resolution.cc
    data/
      README.md

  docs/
    architecture.md
    model_package.md
    build.md
    dependency_policy.md
    embedded.md
    release_packaging.md
    wenet_upgrade_guide.md
```

---

## 5. File responsibilities

### 5.1 Public SDK headers

| File | Goal |
|---|---|
| `include/asr_sdk/asr_engine.h` | Main SDK object. Loads model package once and creates streams. Must not include WeNet or ONNX Runtime headers. |
| `include/asr_sdk/asr_stream.h` | Streaming API: accept PCM, decode one step, get partial/final result, reset. |
| `include/asr_sdk/asr_result.h` | Public result structs: text, score, timestamps, partial/final flag, n-best candidates. |
| `include/asr_sdk/asr_config.h` | Stable SDK config structs. Names should be product-oriented, not direct WeNet flag names. |
| `include/asr_sdk/asr_status.h` | Error codes and status object. No exceptions across C ABI. |
| `include/asr_sdk/c_api.h` | Stable C ABI for embedded systems and FFI. |
| `include/asr_sdk/export.h` | Export macros for Windows/Linux/macOS symbol visibility. |
| `include/asr_sdk/version.h` | SDK version, ABI version, pinned WeNet commit hash, ONNX Runtime version string if available. |

### 5.2 SDK implementation

| File | Goal |
|---|---|
| `src/sdk/asr_engine.cc` | Owns `WenetRuntimeBridge`. Validates config/model package and creates `AsrStream`. |
| `src/sdk/asr_stream.cc` | Owns `WenetStreamAdapter`. Converts public API calls into bridge calls. |
| `src/sdk/asr_config.cc` | Load/merge config from JSON/YAML and explicit config object. |
| `src/sdk/asr_result.cc` | Result copying, JSON serialization, and small helpers. |
| `src/sdk/asr_status.cc` | Status-to-string mapping and error helpers. |
| `src/sdk/c_api.cc` | C ABI handles, lifetime management, error boundary, JSON result buffer. |
| `src/sdk/version.cc` | Compile-time version constants and dependency metadata. |

### 5.3 WeNet bridge layer

| File | Goal |
|---|---|
| `src/wenet_bridge/wenet_runtime_bridge.h/.cc` | Private bridge that loads WeNet runtime resources and creates stream adapters. Includes WeNet headers. |
| `src/wenet_bridge/wenet_stream_adapter.h/.cc` | Per-stream adapter around WeNet feature pipeline and decoder. |
| `src/wenet_bridge/wenet_config_mapper.h/.cc` | Converts SDK config and `sdk_model.json` into WeNet configs/options. |
| `src/wenet_bridge/wenet_result_mapper.h/.cc` | Converts WeNet decode output into `asr_sdk::AsrResult`. |
| `src/wenet_bridge/wenet_error_boundary.h/.cc` | Converts WeNet failures, exceptions, and invalid states into `asr_sdk::Status`. |

Important rule:

```text
Only files under src/wenet_bridge/ may include WeNet headers.
No public SDK header may include or forward declare WeNet types.
```

### 5.4 Model package files

| File | Goal |
|---|---|
| `src/package/model_package.h/.cc` | Represents package paths: ONNX dir, units, words, FST, config, checksums. |
| `src/package/model_package_validator.h/.cc` | Checks all required files before WeNet loads them. |
| `src/package/checksum.h/.cc` | Optional SHA-256 validation for model and graph files. |

### 5.5 CMake files

| File | Goal |
|---|---|
| `cmake/Options.cmake` | Defines SDK build flags. |
| `cmake/OnnxRuntimeDynamic.cmake` | Creates an imported shared `onnxruntime` target from a local ORT package or system path. |
| `cmake/WenetStaticRuntime.cmake` | Adds/pins WeNet and builds selected runtime targets as static/private dependencies. |
| `cmake/SymbolVisibility.cmake` | Hides all internal symbols and exports only SDK API. |
| `cmake/InstallRules.cmake` | Installs SDK headers, SDK library, ONNX Runtime shared lib, docs, licenses. |

---

## 6. Dependency policy

### 6.1 Runtime dependencies

| Dependency | Linkage in production | Owner | Notes |
|---|---:|---|---|
| C++17 runtime | platform | SDK | Keep C++17 for embedded portability. |
| WeNet runtime code | **static/internal** | SDK-vendored | Full source as submodule; build only required runtime targets. |
| ONNX Runtime | **dynamic/bundled** | SDK release package | Do not rely on random system ORT. |
| OpenFst / Kaldi subset | preferably static/internal via WeNet | WeNet dependency | If dynamic is unavoidable, bundle it and keep it private. |
| glog/gflags | preferably static/internal via WeNet | WeNet dependency | Avoid exposing gflags or glog in SDK API. |
| nlohmann/json or yaml-cpp | static or header-only | SDK | Used only for config/package parsing. |
| dr_wav or tiny WAV reader | static/header-only | SDK tools/tests | Not part of core streaming API. |
| GoogleTest | test-only | SDK | Never ship in release. |

### 6.2 Pinning rules

Create:

```text
third_party/wenet.version
third_party/onnxruntime.version
```

Example `third_party/wenet.version`:

```text
repo: https://github.com/alicekenway/wenet
commit: <exact commit hash>
runtime_mode: ONNX
linkage: static into asr_sdk
options:
  ONNX=ON
  TORCH=OFF
  WEBSOCKET=OFF
  GRPC=OFF
  HTTP=OFF
  GRAPH_TOOLS=OFF for release
```

Example `third_party/onnxruntime.version`:

```text
name: onnxruntime
version: <exact version>
linkage: dynamic
linux_soname: libonnxruntime.so
windows_dll: onnxruntime.dll
macos_dylib: libonnxruntime.dylib
sha256: <archive checksum>
source: official ONNX Runtime release or internally approved build
```

### 6.3 Public API dependency rule

Do not expose these in public headers:

```cpp
Ort::Session
Ort::Env
Ort::Value
fst::Fst
fst::SymbolTable
wenet::AsrDecoder
wenet::FeaturePipeline
wenet::DecodeResource
std::shared_ptr<wenet::...>
```

Use PIMPL and C ABI handles:

```cpp
class AsrEngine {
 public:
  static StatusOr<std::unique_ptr<AsrEngine>> Create(const EngineConfig& config);
  ~AsrEngine();

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
```

---

## 7. Build strategy

### 7.1 Build result

Preferred production artifact:

```text
libasr_sdk.so
  contains:
    - our SDK code
    - our WeNet bridge code
    - selected WeNet runtime object code / static libs

  dynamic dependency:
    - libonnxruntime.so
```

Windows equivalent:

```text
asr_sdk.dll
onnxruntime.dll
```

macOS equivalent:

```text
libasr_sdk.dylib
libonnxruntime.dylib
```

### 7.2 Why shared SDK library is preferred

A shared SDK library makes symbol hiding and customer integration easier:

```text
Application
  -> libasr_sdk.so
      -> libonnxruntime.so
```

A static SDK archive is possible, but it pushes more link details to the user:

```text
Application
  -> libasr_sdk.a
  -> libonnxruntime.so
  -> maybe extra private deps
```

For the first industrial SDK release, prefer:

```text
ASR_SDK_BUILD_SHARED=ON
ASR_SDK_STATIC_WENET=ON
ASR_SDK_DYNAMIC_ONNXRUNTIME=ON
```

---

## 8. CMake options

`cmake/Options.cmake`:

```cmake
option(ASR_SDK_BUILD_SHARED "Build SDK as shared library" ON)
option(ASR_SDK_STATIC_WENET "Link WeNet runtime statically into SDK" ON)
option(ASR_SDK_DYNAMIC_ONNXRUNTIME "Use dynamic ONNX Runtime" ON)
option(ASR_SDK_USE_BUNDLED_WENET "Use third_party/wenet submodule" ON)
option(ASR_SDK_USE_BUNDLED_ONNXRUNTIME "Use third_party/onnxruntime package" ON)
option(ASR_SDK_BUILD_TOOLS "Build SDK CLI tools" ON)
option(ASR_SDK_BUILD_TESTS "Build SDK tests" ON)
option(ASR_SDK_BUILD_WENET_TOOLS "Build WeNet tools" OFF)
option(ASR_SDK_ENABLE_GRAPH_TOOLS "Build WeNet/Kaldi graph tools" OFF)
option(ASR_SDK_HIDE_INTERNAL_SYMBOLS "Hide internal symbols" ON)
option(ASR_SDK_ENABLE_SANITIZERS "Enable sanitizer flags" OFF)
```

Codex requirement:

```text
The production preset must set:
  ASR_SDK_STATIC_WENET=ON
  ASR_SDK_DYNAMIC_ONNXRUNTIME=ON
  ASR_SDK_HIDE_INTERNAL_SYMBOLS=ON
  ASR_SDK_BUILD_WENET_TOOLS=OFF
  ASR_SDK_ENABLE_GRAPH_TOOLS=OFF
```

---

## 9. Dynamic ONNX Runtime CMake target

Create `cmake/OnnxRuntimeDynamic.cmake`.

Goal:

```text
Create an imported shared target named onnxruntime.
The target must point to a known bundled ORT library, not a random system library by default.
```

Example:

```cmake
if(NOT ASR_SDK_DYNAMIC_ONNXRUNTIME)
  message(FATAL_ERROR "This plan expects dynamic ONNX Runtime. Static ORT is not implemented in MVP.")
endif()

set(ASR_SDK_ONNXRUNTIME_ROOT
    "${CMAKE_SOURCE_DIR}/third_party/onnxruntime"
    CACHE PATH "Path to ONNX Runtime package")

if(WIN32)
  set(ORT_LIB "${ASR_SDK_ONNXRUNTIME_ROOT}/lib/onnxruntime.lib")
  set(ORT_RUNTIME "${ASR_SDK_ONNXRUNTIME_ROOT}/bin/onnxruntime.dll")
elseif(APPLE)
  set(ORT_LIB "${ASR_SDK_ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib")
  set(ORT_RUNTIME "${ORT_LIB}")
else()
  set(ORT_LIB "${ASR_SDK_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
  set(ORT_RUNTIME "${ORT_LIB}")
endif()

if(NOT EXISTS "${ASR_SDK_ONNXRUNTIME_ROOT}/include/onnxruntime_c_api.h")
  message(FATAL_ERROR "ONNX Runtime headers not found: ${ASR_SDK_ONNXRUNTIME_ROOT}/include")
endif()

if(NOT EXISTS "${ORT_LIB}")
  message(FATAL_ERROR "ONNX Runtime shared library not found: ${ORT_LIB}")
endif()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ORT_RUNTIME}"
  INTERFACE_INCLUDE_DIRECTORIES "${ASR_SDK_ONNXRUNTIME_ROOT}/include"
)

if(WIN32)
  set_target_properties(onnxruntime PROPERTIES
    IMPORTED_IMPLIB "${ORT_LIB}"
  )
endif()
```

Notes for Codex:

```text
- Do not use find_package(ONNXRuntime) as the default production path.
- A system ORT path can be added later, but release builds must use a pinned package.
- Copy the ORT shared library into the SDK install/package directory.
```

---

## 10. Static WeNet runtime CMake integration

Create `cmake/WenetStaticRuntime.cmake`.

Goal:

```text
Use the full pinned WeNet submodule.
Build only required WeNet ONNX runtime targets.
Link those static targets privately into asr_sdk.
Ensure WeNet uses the imported dynamic onnxruntime target.
```

### 10.1 Preferred direct integration

Conceptual CMake:

```cmake
if(NOT ASR_SDK_STATIC_WENET)
  message(FATAL_ERROR "Production build requires static WeNet runtime")
endif()

set(WENET_ROOT "${CMAKE_SOURCE_DIR}/third_party/wenet" CACHE PATH "Path to WeNet source")

if(NOT EXISTS "${WENET_ROOT}/runtime")
  message(FATAL_ERROR "WeNet submodule missing. Run: git submodule update --init --recursive")
endif()

# WeNet build mode.
set(ONNX ON CACHE BOOL "Build WeNet ONNX runtime" FORCE)
set(TORCH OFF CACHE BOOL "Disable WeNet Torch runtime" FORCE)
set(WEBSOCKET OFF CACHE BOOL "Disable WeNet websocket" FORCE)
set(GRPC OFF CACHE BOOL "Disable WeNet grpc" FORCE)
set(HTTP OFF CACHE BOOL "Disable WeNet HTTP" FORCE)
set(GRAPH_TOOLS ${ASR_SDK_ENABLE_GRAPH_TOOLS} CACHE BOOL "Build graph tools" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable WeNet tests" FORCE)

# Important:
# The imported target `onnxruntime` must already exist from OnnxRuntimeDynamic.cmake.
# WeNet decoder target should link to this dynamic imported target when ONNX=ON.

add_subdirectory(
  "${WENET_ROOT}/runtime/onnxruntime"
  "${CMAKE_BINARY_DIR}/third_party_build/wenet_onnxruntime"
)
```

### 10.2 If upstream CMake is too invasive

If direct `add_subdirectory()` pulls in tools, global flags, downloads, or target conflicts, Codex should create a private minimal static target while still compiling from the pinned WeNet submodule.

Create target:

```text
asr_sdk_wenet_runtime_static
```

It should compile/link the required WeNet runtime libraries and sources from the submodule, not copied files. This target is private to our SDK.

Policy:

```text
Allowed:
  Build selected WeNet sources from third_party/wenet by CMake target.
  Add small documented patches under patches/wenet/ if necessary.
  Keep a script to reapply patches after submodule update.

Not allowed:
  Copy selected WeNet files into src/ as our own fork.
  Modify third_party/wenet manually without patches.
  Expose WeNet target names to SDK users.
```

### 10.3 Target names and adaptation

WeNet upstream target names may change. Codex should keep all target-name assumptions inside:

```text
cmake/WenetStaticRuntime.cmake
```

The rest of the SDK should link only to our internal alias:

```cmake
add_library(asr_sdk::wenet_runtime_static ALIAS <actual-wenet-target-or-bundle>)
```

Then top-level SDK links:

```cmake
target_link_libraries(asr_sdk
  PRIVATE
    asr_sdk::wenet_runtime_static
    onnxruntime
)
```

---

## 11. Top-level CMake skeleton

```cmake
cmake_minimum_required(VERSION 3.18)
project(wenet_static_ort_dynamic_sdk LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/Options.cmake)
include(cmake/OnnxRuntimeDynamic.cmake)
include(cmake/WenetStaticRuntime.cmake)
include(cmake/SymbolVisibility.cmake)

if(ASR_SDK_BUILD_SHARED)
  add_library(asr_sdk SHARED)
else()
  add_library(asr_sdk STATIC)
endif()

target_sources(asr_sdk PRIVATE
  src/sdk/asr_engine.cc
  src/sdk/asr_stream.cc
  src/sdk/asr_config.cc
  src/sdk/asr_result.cc
  src/sdk/asr_status.cc
  src/sdk/c_api.cc
  src/sdk/version.cc

  src/wenet_bridge/wenet_runtime_bridge.cc
  src/wenet_bridge/wenet_stream_adapter.cc
  src/wenet_bridge/wenet_config_mapper.cc
  src/wenet_bridge/wenet_result_mapper.cc
  src/wenet_bridge/wenet_error_boundary.cc

  src/package/model_package.cc
  src/package/model_package_validator.cc
  src/package/checksum.cc

  src/audio/wav_reader.cc
  src/audio/resampler.cc
  src/audio/ring_buffer.cc

  src/utils/file_utils.cc
  src/utils/logging.cc
  src/utils/timer.cc
  src/utils/metrics.cc
)

target_include_directories(asr_sdk
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${WENET_ROOT}/runtime/core
    ${WENET_ROOT}/runtime/onnxruntime
)

target_link_libraries(asr_sdk
  PRIVATE
    asr_sdk::wenet_runtime_static
    onnxruntime
)

if(ASR_SDK_HIDE_INTERNAL_SYMBOLS)
  asr_sdk_apply_hidden_visibility(asr_sdk)
endif()

include(cmake/InstallRules.cmake)
```

---

## 12. Symbol visibility

Because WeNet is statically linked into `libasr_sdk.so`, WeNet symbols could accidentally become visible to SDK users. Hide them.

Create `include/asr_sdk/export.h`:

```cpp
#pragma once

#if defined(_WIN32)
  #if defined(ASR_SDK_BUILDING_LIBRARY)
    #define ASR_SDK_API __declspec(dllexport)
  #else
    #define ASR_SDK_API __declspec(dllimport)
  #endif
#else
  #define ASR_SDK_API __attribute__((visibility("default")))
#endif
```

CMake visibility helper:

```cmake
function(asr_sdk_apply_hidden_visibility target)
  set_target_properties(${target} PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    C_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )

  target_compile_definitions(${target} PRIVATE ASR_SDK_BUILDING_LIBRARY=1)

  if(UNIX AND NOT APPLE)
    target_link_options(${target} PRIVATE
      "LINKER:--exclude-libs,ALL"
    )
  endif()
endfunction()
```

Linux version script, optional but recommended:

```text
ASR_SDK_1.0 {
  global:
    asr_sdk_*;
  local:
    *;
};
```

Codex acceptance check:

```bash
nm -D libasr_sdk.so | grep wenet      # should be empty or only intentionally exported symbols
nm -D libasr_sdk.so | grep Ort        # should be empty or minimal dynamic references only
nm -D libasr_sdk.so | grep asr_sdk_   # should show public C ABI symbols
```

---

## 13. Runtime path and release packaging

### 13.1 Linux RPATH

Set the SDK shared library to find ONNX Runtime next to itself:

```cmake
set_target_properties(asr_sdk PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)
```

For a nested layout:

```text
bin/app
lib/libasr_sdk.so
lib/libonnxruntime.so
```

Use:

```cmake
set_target_properties(asr_sdk PROPERTIES
  INSTALL_RPATH "$ORIGIN"
)
```

For app executables installed to `bin/`:

```cmake
set_target_properties(asr_stream_file PROPERTIES
  INSTALL_RPATH "$ORIGIN/../lib"
)
```

### 13.2 Release package layout

Linux:

```text
asr_sdk_release_linux_x86_64/
  include/
    asr_sdk/*.h
  lib/
    libasr_sdk.so
    libonnxruntime.so
  bin/
    asr_stream_file
    benchmark
    inspect_package
  models/
    README.md
  licenses/
    LICENSE_ASR_SDK
    LICENSE_WENET
    LICENSE_ONNXRUNTIME
    LICENSE_OPENFST
    THIRD_PARTY_NOTICES.txt
  docs/
    build.md
    model_package.md
    embedded.md
```

Windows:

```text
asr_sdk_release_windows_x64/
  include/
  lib/
    asr_sdk.lib
    onnxruntime.lib
  bin/
    asr_sdk.dll
    onnxruntime.dll
    asr_stream_file.exe
  licenses/
  docs/
```

macOS:

```text
asr_sdk_release_macos_arm64/
  include/
  lib/
    libasr_sdk.dylib
    libonnxruntime.dylib
  bin/
  licenses/
  docs/
```

### 13.3 Dependency verification command

Linux:

```bash
ldd lib/libasr_sdk.so
readelf -d lib/libasr_sdk.so | grep -E 'NEEDED|RPATH|RUNPATH'
```

Expected:

```text
NEEDED: libonnxruntime.so
RUNPATH/RPATH: $ORIGIN
No dependency on libwenet*.so
```

Windows:

```powershell
dumpbin /DEPENDENTS bin\asr_sdk.dll
```

Expected:

```text
onnxruntime.dll
No wenet_runtime.dll
```

macOS:

```bash
otool -L lib/libasr_sdk.dylib
```

Expected:

```text
@rpath/libonnxruntime.dylib
No libwenet*.dylib
```

---

## 14. Model package contract

### 14.1 Required model directory

Use this package layout:

```text
model_dir/
  sdk_model.json
  onnx/
    encoder.onnx
    ctc.onnx
    decoder.onnx
  units.txt
  words.txt              # required for WFST / TLG decoding
  TLG.fst                # required for WFST / n-gram decoding
  checksums.sha256       # optional but recommended
```

Important MVP note:

```text
Even if the product only wants CTC + WFST and no attention rescoring,
include decoder.onnx in MVP because WeNet's ONNX export/runtime path commonly expects:
  encoder.onnx
  ctc.onnx
  decoder.onnx

Disable attention rescoring by setting:
  rescoring_weight = 0.0
  ctc_weight = 1.0
```

A later optimization may patch the bridge for a CTC-only ONNX package, but that is **not MVP**.

### 14.2 Example `sdk_model.json`

```json
{
  "sdk_model_version": 1,
  "backend": "wenet_onnxruntime_static_wenet_dynamic_ort",

  "audio": {
    "sample_rate": 16000,
    "num_bins": 80,
    "feature_type": "kaldi"
  },

  "wenet": {
    "onnx_dir": "onnx",
    "unit_path": "units.txt",
    "dict_path": "words.txt",
    "fst_path": "TLG.fst"
  },

  "decode": {
    "chunk_size": 16,
    "num_left_chunks": -1,
    "ctc_weight": 1.0,
    "rescoring_weight": 0.0,
    "reverse_weight": 0.0,

    "beam": 16.0,
    "lattice_beam": 10.0,
    "max_active": 7000,
    "min_active": 200,
    "acoustic_scale": 1.0,
    "blank_id": 0,
    "blank_skip_thresh": 0.98,
    "blank_scale": 1.0,
    "length_penalty": 0.0,
    "nbest": 1
  },

  "postprocess": {
    "language_type": "en",
    "lowercase": true,
    "enable_timestamp": false
  },

  "runtime": {
    "onnxruntime_threads": 2,
    "enable_continuous_decoding": true
  }
}
```

### 14.3 Validator requirements

`ModelPackageValidator` must check:

```text
[ ] sdk_model.json exists and version is supported.
[ ] onnx/encoder.onnx exists.
[ ] onnx/ctc.onnx exists.
[ ] onnx/decoder.onnx exists for MVP.
[ ] units.txt exists.
[ ] If fst_path exists, dict_path must also exist.
[ ] If WFST mode enabled, TLG.fst exists.
[ ] chunk_size is present.
[ ] num_left_chunks is present.
[ ] sample_rate is supported.
[ ] checksums match if checksums.sha256 exists.
```

---

## 15. Public C++ API

### 15.1 Example usage

```cpp
#include <asr_sdk/asr_engine.h>
#include <asr_sdk/asr_stream.h>

int main() {
  asr_sdk::EngineConfig config;
  config.model_dir = "./model_dir";
  config.num_threads = 2;
  config.enable_continuous_decoding = true;

  auto engine_or = asr_sdk::AsrEngine::Create(config);
  if (!engine_or.ok()) {
    std::cerr << engine_or.status().Message() << "\n";
    return 1;
  }

  auto engine = std::move(engine_or.value());
  auto stream = engine->CreateStream();

  // Feed 100 ms chunks.
  for (const auto& chunk : ReadPcm16Chunks("test.wav", 100)) {
    stream->AcceptPcm16(chunk.samples.data(), chunk.samples.size(), chunk.sample_rate);

    while (stream->DecodeReady()) {
      auto status = stream->Decode();
      if (!status.ok()) {
        std::cerr << status.Message() << "\n";
        return 1;
      }

      auto partial = stream->GetResult();
      if (!partial.text.empty()) {
        std::cout << "[partial] " << partial.text << "\n";
      }
    }
  }

  stream->SetInputFinished();
  while (stream->DecodeReady()) {
    stream->Decode();
  }

  auto final_result = stream->GetFinalResult();
  std::cout << "[final] " << final_result.text << "\n";
  return 0;
}
```

### 15.2 Public classes

```cpp
namespace asr_sdk {

struct EngineConfig {
  std::string model_dir;
  int num_threads = 1;
  bool enable_continuous_decoding = true;
  bool enable_timestamps = false;
  bool enable_nbest = false;
  int nbest = 1;
};

struct TokenResult {
  std::string token;
  int token_id = -1;
  float start_ms = -1.0f;
  float end_ms = -1.0f;
  float confidence = 0.0f;
};

struct NBestResult {
  std::string text;
  float score = 0.0f;
  std::vector<TokenResult> tokens;
};

struct AsrResult {
  std::string text;
  bool is_final = false;
  float confidence = 0.0f;
  std::vector<TokenResult> tokens;
  std::vector<NBestResult> nbest;
};

class AsrStream {
 public:
  virtual ~AsrStream() = default;

  virtual Status AcceptPcm16(const int16_t* samples,
                             size_t num_samples,
                             int sample_rate) = 0;

  virtual bool DecodeReady() const = 0;
  virtual Status Decode() = 0;

  virtual AsrResult GetResult() const = 0;
  virtual AsrResult GetFinalResult() = 0;

  virtual Status SetInputFinished() = 0;
  virtual Status Reset() = 0;
};

class AsrEngine {
 public:
  static StatusOr<std::unique_ptr<AsrEngine>> Create(const EngineConfig& config);

  virtual ~AsrEngine() = default;
  virtual StatusOr<std::unique_ptr<AsrStream>> CreateStream() = 0;
};

}  // namespace asr_sdk
```

---

## 16. C ABI

The C ABI should be stable and avoid C++ types.

```c
typedef struct AsrSdkEngine AsrSdkEngine;
typedef struct AsrSdkStream AsrSdkStream;

int asr_sdk_create_engine(const char* model_dir, AsrSdkEngine** out_engine);
void asr_sdk_destroy_engine(AsrSdkEngine* engine);

int asr_sdk_create_stream(AsrSdkEngine* engine, AsrSdkStream** out_stream);
void asr_sdk_destroy_stream(AsrSdkStream* stream);

int asr_sdk_accept_pcm16(
    AsrSdkStream* stream,
    const int16_t* samples,
    int num_samples,
    int sample_rate);

int asr_sdk_decode(AsrSdkStream* stream);
int asr_sdk_decode_ready(AsrSdkStream* stream);
int asr_sdk_set_input_finished(AsrSdkStream* stream);
int asr_sdk_reset_stream(AsrSdkStream* stream);

const char* asr_sdk_get_result_json(AsrSdkStream* stream);
const char* asr_sdk_get_final_result_json(AsrSdkStream* stream);

int asr_sdk_last_error_code(void* handle);
const char* asr_sdk_last_error_message(void* handle);

const char* asr_sdk_version(void);
const char* asr_sdk_build_info_json(void);
```

C ABI result JSON example:

```json
{
  "text": "turn on the light",
  "is_final": true,
  "confidence": 0.87,
  "nbest": [
    {
      "text": "turn on the light",
      "score": -12.4
    }
  ],
  "tokens": [
    {
      "token": "turn",
      "start_ms": 120,
      "end_ms": 300,
      "confidence": 0.91
    }
  ]
}
```

---

## 17. WeNet bridge design

### 17.1 Bridge responsibilities

`WenetRuntimeBridge`:

```text
- Own resolved model package paths.
- Own mapped WeNet decode/resource config.
- Own shared resources that can be reused across streams.
- Validate that ONNX Runtime dynamic library is available.
- Create WenetStreamAdapter instances.
```

`WenetStreamAdapter`:

```text
- Own per-stream WeNet objects.
- Accept PCM and pass it to WeNet frontend/decoder.
- Run Decode one step/chunk.
- Retrieve partial/final result.
- Reset for next utterance or continuous decoding.
```

`WenetConfigMapper`:

```text
SDK config + sdk_model.json
  -> WeNet feature config
  -> WeNet decode options
  -> WeNet resource paths
```

`WenetResultMapper`:

```text
WeNet result object / result JSON / decode result vector
  -> asr_sdk::AsrResult
```

### 17.2 Adapter should isolate WeNet churn

All WeNet-specific usage must stay in:

```text
src/wenet_bridge/
```

When updating WeNet, expected code changes should be limited to:

```text
cmake/WenetStaticRuntime.cmake
src/wenet_bridge/*
tests/integration/test_compare_with_wenet_decoder_main.cc
```

### 17.3 WeNet C API note

WeNet has a C-style API in its runtime tree, but its upstream CMake may build it as a shared library. Since our production rule is **no dynamic WeNet dependency**, do not expose or link against a separate `libwenet_api.so` in the release.

Allowed options:

```text
Option A, preferred production:
  Use WeNet internal C++ runtime classes inside src/wenet_bridge.

Option B, temporary development shortcut:
  Compile WeNet's API implementation into a private object/static target and hide its symbols.
  Do not ship libwenet_api.so.

Option C, debug only:
  Compare against WeNet decoder_main or libwenet_api.so outside the SDK.
  Do not make this a production dependency.
```

---

## 18. Streaming behavior

### 18.1 Decode loop contract

```text
AcceptPcm16()
  -> pushes samples into the WeNet-backed stream/frontend.

DecodeReady()
  -> true when enough audio/features are available or input is finished.

Decode()
  -> performs one bounded decode step.

GetResult()
  -> returns current partial result.

SetInputFinished()
  -> marks end of utterance/input.

GetFinalResult()
  -> finalizes decoder and returns final result.
```

### 18.2 Audio callback rule

Do not run ONNX inference or WFST decoding in a real-time audio callback.

Good embedded pattern:

```text
Audio callback thread:
  copy PCM to bounded ring buffer

Decode worker thread:
  pop PCM from ring buffer
  call AcceptPcm16()
  call Decode()
  publish partial/final result
```

`examples/embedded/audio_callback_pattern.cc` should demonstrate this.

---

## 19. Tools

### 19.1 `asr_stream_file`

Command:

```bash
asr_stream_file \
  --model_dir ./model_dir \
  --wav ./test.wav \
  --chunk_ms 100 \
  --print_partial true
```

Output:

```text
[partial] hello
[partial] hello world
[final] hello world
RTF: 0.42
first_partial_latency_ms: 180
onnxruntime: dynamic
wenet_linkage: static
```

### 19.2 `inspect_package`

Command:

```bash
inspect_package --model_dir ./model_dir
```

Output should include:

```text
sdk_model_version: 1
backend: wenet_onnxruntime_static_wenet_dynamic_ort
onnx_dir: model_dir/onnx
encoder.onnx: ok
ctc.onnx: ok
decoder.onnx: ok
units.txt: ok
TLG.fst: ok
words.txt: ok
chunk_size: 16
num_left_chunks: -1
onnxruntime_linkage: dynamic
wenet_linkage: static
```

### 19.3 `print_build_info`

Command:

```bash
print_build_info
```

Output JSON:

```json
{
  "sdk_version": "0.1.0",
  "abi_version": 1,
  "wenet": {
    "commit": "<commit>",
    "linkage": "static"
  },
  "onnxruntime": {
    "version": "<version>",
    "linkage": "dynamic"
  },
  "build": {
    "compiler": "gcc/clang/msvc",
    "cxx_standard": "17",
    "symbol_visibility": "hidden"
  }
}
```

---

## 20. Development phases for Codex

### Phase 0 — Dependency scaffolding

Deliverables:

```text
third_party/wenet.version
third_party/onnxruntime.version
cmake/Options.cmake
cmake/OnnxRuntimeDynamic.cmake
cmake/WenetStaticRuntime.cmake
cmake/SymbolVisibility.cmake
scripts/add_wenet_submodule.sh
scripts/fetch_onnxruntime.sh
```

Tasks:

```text
1. Add build options.
2. Add imported dynamic ONNX Runtime target.
3. Add WeNet submodule detection.
4. Add initial static WeNet integration target or alias.
5. Add configure-time printout showing:
   - WeNet root
   - ONNX Runtime root
   - WeNet linkage = static
   - ONNX Runtime linkage = dynamic
```

Acceptance:

```bash
cmake -B build \
  -DASR_SDK_STATIC_WENET=ON \
  -DASR_SDK_DYNAMIC_ONNXRUNTIME=ON
```

Configure must fail clearly if:

```text
third_party/wenet is missing
third_party/onnxruntime is missing
libonnxruntime.so/dll/dylib is missing
ONNX Runtime headers are missing
```

---

### Phase 1 — SDK skeleton

Deliverables:

```text
include/asr_sdk/*.h
src/sdk/*.cc
src/package/model_package.*
tools/print_build_info.cc
```

Tasks:

```text
1. Implement Status and StatusOr.
2. Implement EngineConfig.
3. Implement AsrEngine and AsrStream skeletons.
4. Implement C ABI handle skeleton.
5. Implement build info function.
6. Do not call WeNet yet.
```

Acceptance:

```bash
cmake --build build -j
./build/tools/print_build_info
```

Expected:

```text
wenet_linkage: static
onnxruntime_linkage: dynamic
```

---

### Phase 2 — Model package loader and validator

Deliverables:

```text
src/package/model_package.h/.cc
src/package/model_package_validator.h/.cc
tools/inspect_package.cc
tests/unit/test_model_package_validator.cc
```

Tasks:

```text
1. Parse sdk_model.json.
2. Resolve relative paths.
3. Validate required files.
4. Validate decode config constraints.
5. Validate WFST mode requirements.
6. Add optional checksum validation.
```

Acceptance:

```bash
inspect_package --model_dir ./test_model
```

must report all required files and config values.

---

### Phase 3 — WeNet bridge load

Deliverables:

```text
src/wenet_bridge/wenet_runtime_bridge.*
src/wenet_bridge/wenet_config_mapper.*
tests/integration/test_wenet_bridge_load.cc
```

Tasks:

```text
1. Include WeNet headers only in src/wenet_bridge.
2. Map SDK model package to WeNet paths/options.
3. Construct WeNet shared resource/model objects.
4. Return clear SDK Status on failure.
```

Acceptance:

```text
test_wenet_bridge_load loads a valid model package without decoding audio.
Invalid package returns a structured SDK error, not a crash.
```

---

### Phase 4 — Streaming decode through WeNet

Deliverables:

```text
src/wenet_bridge/wenet_stream_adapter.*
src/wenet_bridge/wenet_result_mapper.*
tools/asr_stream_file.cc
tests/integration/test_decode_wav_prefix.cc
```

Tasks:

```text
1. Create stream-local WeNet frontend/decoder.
2. Accept PCM16 input.
3. Decode in streaming chunks.
4. Return partial result.
5. Finalize result after SetInputFinished().
6. Map WeNet result into AsrResult.
```

Acceptance:

```bash
asr_stream_file --model_dir ./model_dir --wav ./test.wav --chunk_ms 100
```

Output must show partial/final result.

---

### Phase 5 — WFST / n-gram decoding

Deliverables:

```text
tests/integration/test_decode_wav_wfst.cc
docs/model_package.md
scripts/build_tlg_with_wenet.sh
```

Tasks:

```text
1. Support fst_path and dict_path in sdk_model.json.
2. Map WFST decode options.
3. Verify TLG.fst and words.txt exist.
4. Decode WAV with WFST mode.
5. Compare against WeNet decoder_main where possible.
```

Acceptance:

```text
With fst_path + dict_path:
  decoder uses WFST/n-gram path.

Without fst_path:
  decoder falls back to CTC prefix path.
```

---

### Phase 6 — C ABI

Deliverables:

```text
include/asr_sdk/c_api.h
src/sdk/c_api.cc
examples/c/simple_c_api.c
tests/unit/test_c_api_lifetime.cc
```

Tasks:

```text
1. Implement engine/stream handles.
2. Implement PCM accept/decode/result functions.
3. Implement last-error retrieval.
4. Ensure result JSON buffer lifetime is documented.
5. Ensure no C++ exceptions escape C ABI.
```

Acceptance:

```bash
examples/c/simple_c_api ./model_dir ./test.wav
```

---

### Phase 7 — Packaging and dependency verification

Deliverables:

```text
cmake/InstallRules.cmake
scripts/package_release.sh
scripts/verify_release_deps.sh
docs/release_packaging.md
tests/integration/test_dynamic_ort_resolution.cc
```

Tasks:

```text
1. Install headers.
2. Install libasr_sdk.
3. Install ONNX Runtime shared library.
4. Install tools.
5. Install licenses/notices.
6. Verify no libwenet*.so dependency exists.
7. Verify libonnxruntime dynamic dependency exists.
8. Verify RPATH/RUNPATH resolves ORT locally.
```

Acceptance:

Linux:

```bash
scripts/package_release.sh
scripts/verify_release_deps.sh release/
```

Expected:

```text
PASS: libasr_sdk.so found
PASS: libonnxruntime.so found
PASS: libasr_sdk.so depends on libonnxruntime.so
PASS: libasr_sdk.so does not depend on libwenet*.so
PASS: internal symbols hidden
PASS: sample WAV decodes from packaged release
```

---

### Phase 8 — Embedded pattern and performance

Deliverables:

```text
examples/embedded/audio_callback_pattern.cc
tools/benchmark.cc
docs/embedded.md
```

Tasks:

```text
1. Add bounded audio ring buffer example.
2. Add decode worker example.
3. Add benchmark metrics.
4. Add memory and latency report.
5. Add cross-compilation toolchain skeleton.
```

Benchmark should report:

```text
audio_sec
wall_sec
RTF
first_partial_latency_ms
final_latency_ms
feature_decode_ms if available
onnx_decode_ms if available
decoder_ms if available
peak_rss_mb
wenet_linkage
onnxruntime_linkage
```

---

## 21. Testing plan

### 21.1 Unit tests

| Test | Goal |
|---|---|
| `test_config_mapper.cc` | SDK config maps correctly into WeNet-related config. |
| `test_model_package_validator.cc` | Required package files and invalid configurations are detected early. |
| `test_result_mapper.cc` | WeNet result object/JSON converts correctly into SDK result. |
| `test_c_api_lifetime.cc` | Engine/stream/result buffer lifetimes are safe. |
| `test_status.cc` | Error codes and messages are stable. |

### 21.2 Integration tests

| Test | Goal |
|---|---|
| `test_wenet_bridge_load.cc` | Load static WeNet-backed runtime with dynamic ONNX Runtime. |
| `test_decode_wav_prefix.cc` | Decode known WAV without WFST. |
| `test_decode_wav_wfst.cc` | Decode known WAV with TLG.fst and words.txt. |
| `test_compare_with_wenet_decoder_main.cc` | Compare SDK result against WeNet CLI baseline for regression. |
| `test_dynamic_ort_resolution.cc` | Verify SDK loads bundled ONNX Runtime, not a random system copy. |

### 21.3 Packaging tests

```bash
ldd release/lib/libasr_sdk.so
readelf -d release/lib/libasr_sdk.so
nm -D release/lib/libasr_sdk.so
release/bin/asr_stream_file --model_dir release/models/test_model --wav tests/data/test.wav
```

Packaging must confirm:

```text
[ ] ONNX Runtime is dynamic.
[ ] ONNX Runtime is bundled.
[ ] WeNet is not a dynamic dependency.
[ ] Public symbols are only SDK symbols.
[ ] The release package can decode a known WAV on a clean machine/container.
```

---

## 22. CI plan

### 22.1 CI jobs

```text
linux-x86_64-debug
linux-x86_64-release
linux-x86_64-package-test
linux-aarch64-cross-compile
windows-x64-release       # later
macos-arm64-release       # later
```

### 22.2 CI steps

```text
1. Checkout repository with submodules.
2. Fetch/pin ONNX Runtime package.
3. Configure CMake.
4. Build SDK.
5. Build tools.
6. Run unit tests.
7. Run integration tests if model fixture is available.
8. Package release.
9. Verify dynamic dependencies.
10. Upload artifacts.
```

### 22.3 CI guardrails

CI should fail if:

```text
- WeNet submodule commit differs from third_party/wenet.version.
- ONNX Runtime package checksum differs from third_party/onnxruntime.version.
- libasr_sdk dynamically links libwenet*.so.
- libasr_sdk does not dynamically link libonnxruntime.so/dll/dylib.
- Public headers include WeNet or ONNX Runtime headers.
- Exported symbols include unexpected WeNet internals.
```

---

## 23. Embedded deployment notes

### 23.1 Preferred embedded release shape

```text
/opt/asr_sdk/
  lib/
    libasr_sdk.so
    libonnxruntime.so
  models/
    command_model/
      sdk_model.json
      onnx/encoder.onnx
      onnx/ctc.onnx
      onnx/decoder.onnx
      units.txt
      words.txt
      TLG.fst
  bin/
    asr_stream_file
    benchmark
```

### 23.2 Audio callback pattern

```cpp
void AudioCallback(const int16_t* input, size_t n) {
  // Copy only. No ONNX inference, no WFST decoding, no logging flood.
  ring_buffer.Push(input, n);
}

void DecodeWorker(asr_sdk::AsrStream* stream) {
  std::vector<int16_t> chunk;
  while (running) {
    if (ring_buffer.Pop(&chunk)) {
      stream->AcceptPcm16(chunk.data(), chunk.size(), 16000);
      while (stream->DecodeReady()) {
        stream->Decode();
        Publish(stream->GetResult());
      }
    }
  }
}
```

### 23.3 Industrial constraints

```text
[ ] No heavy allocation in audio callback.
[ ] Bound audio queue size.
[ ] Bound result JSON buffer size or document maximum.
[ ] Avoid writing logs to flash continuously.
[ ] Validate model package before accepting live audio.
[ ] Provide watchdog-friendly decode loop.
[ ] Provide clear error if ONNX Runtime shared library is missing.
[ ] Provide build info API for field diagnostics.
```

---

## 24. WeNet upgrade policy

Create `docs/wenet_upgrade_guide.md`.

Upgrade checklist:

```text
1. Create branch: upgrade/wenet-<date>.
2. Update third_party/wenet submodule to a specific commit.
3. Reapply patches from patches/wenet/.
4. Build WeNet runtime sanity target.
5. Build SDK.
6. Run unit tests.
7. Run bridge-load integration test.
8. Run prefix decode golden tests.
9. Run WFST decode golden tests.
10. Compare against WeNet decoder_main output.
11. Compare RTF and memory against previous pinned commit.
12. Inspect exported symbols.
13. Verify no dynamic libwenet dependency.
14. Update third_party/wenet.version.
15. Update release notes.
```

Do not update WeNet by tracking `main` implicitly.

---

## 25. ONNX Runtime upgrade policy

Create `docs/onnxruntime_upgrade_guide.md`.

Upgrade checklist:

```text
1. Download approved ONNX Runtime package for target platform.
2. Verify checksum.
3. Update third_party/onnxruntime.version.
4. Build SDK.
5. Run ONNX model load test.
6. Run WAV golden tests.
7. Run package dependency verification.
8. Compare RTF and memory.
9. Check CPU instruction compatibility on target device.
10. Update release notes.
```

Do not rely on `/usr/lib/libonnxruntime.so` in production releases.

---

## 26. Risks and mitigations

| Risk | Mitigation |
|---|---|
| WeNet CMake pulls too many targets | Keep all WeNet CMake assumptions in `WenetStaticRuntime.cmake`; add minimal private bundle target if necessary. |
| WeNet symbols leak from `libasr_sdk.so` | Use hidden visibility, `--exclude-libs,ALL`, and optional linker version script. |
| Dynamic ONNX Runtime missing at customer site | Bundle ORT and set RPATH/RUNPATH; add startup diagnostic. |
| Random system ONNX Runtime is loaded | Prefer `$ORIGIN` RPATH and verify with `ldd`/`readelf`; package ORT next to SDK. |
| WeNet internal API changes | Keep WeNet usage inside `src/wenet_bridge/`; pin commit; add upgrade checklist. |
| Model package mismatches export config | Validate `chunk_size`, `num_left_chunks`, required ONNX files, units, FST/dict files. |
| C++ ABI leaks to users | Provide stable C ABI; keep C++ API clean; avoid exposing STL-heavy objects across dynamic boundaries where possible. |
| Full static ORT requested later | Treat as Phase 2 optimization with reduced operator build; keep SDK API unchanged. |

---

## 27. Codex implementation order

Use this commit order:

```text
Commit 1: Add repository skeleton and public headers.
Commit 2: Add CMake options, dynamic ONNX Runtime imported target, and version files.
Commit 3: Add WeNet submodule detection and static runtime CMake integration.
Commit 4: Add symbol visibility and build info tool.
Commit 5: Implement Status/StatusOr and SDK skeleton.
Commit 6: Implement model package loader and validator.
Commit 7: Implement WeNet config mapper.
Commit 8: Implement WenetRuntimeBridge load-only path.
Commit 9: Implement WenetStreamAdapter decode path.
Commit 10: Implement result mapper.
Commit 11: Implement asr_stream_file and inspect_package tools.
Commit 12: Implement C ABI.
Commit 13: Add tests.
Commit 14: Add install/package rules and dependency verification scripts.
Commit 15: Add embedded example and benchmark tool.
Commit 16: Write docs: build, model package, dependency policy, embedded deployment.
```

---

## 28. Acceptance criteria

A release candidate is acceptable when all of these pass:

```text
[ ] SDK builds with ASR_SDK_STATIC_WENET=ON.
[ ] SDK builds with ASR_SDK_DYNAMIC_ONNXRUNTIME=ON.
[ ] Public headers do not include WeNet headers.
[ ] Public headers do not include ONNX Runtime headers.
[ ] `libasr_sdk.so` does not depend on `libwenet*.so`.
[ ] `libasr_sdk.so` does depend on bundled `libonnxruntime.so`.
[ ] `ldd` or equivalent confirms ORT resolves from release package.
[ ] Exported symbols are only SDK symbols.
[ ] `inspect_package` validates a model package.
[ ] `asr_stream_file` decodes a known WAV.
[ ] WFST mode works with `TLG.fst` and `words.txt`.
[ ] Prefix mode works without `TLG.fst`.
[ ] C ABI example works.
[ ] Benchmark reports RTF and dependency linkage.
[ ] Release package includes third-party licenses and notices.
```

---

## 29. References for implementation

Use these during implementation and keep exact dependency versions pinned in the repository:

```text
WeNet repository:
  https://github.com/alicekenway/wenet

WeNet ONNX runtime README:
  https://github.com/alicekenway/wenet/blob/main/runtime/onnxruntime/README.md

WeNet runtime C API:
  https://github.com/alicekenway/wenet/blob/main/runtime/core/api/wenet_api.h

WeNet decoder CMake:
  https://github.com/alicekenway/wenet/blob/main/runtime/core/decoder/CMakeLists.txt

WeNet Kaldi subset CMake:
  https://github.com/alicekenway/wenet/blob/main/runtime/core/kaldi/CMakeLists.txt

ONNX Runtime docs:
  https://onnxruntime.ai/docs/

OpenFst:
  https://www.openfst.org/
```

---

## 30. Summary decision

The SDK should use this dependency shape:

```text
Full WeNet source as pinned submodule:
  yes

Manually copied mini-WeNet source tree:
  no, not for MVP

WeNet runtime linkage in product SDK:
  static/internal

ONNX Runtime linkage in product SDK:
  dynamic/bundled

Public SDK dependency on WeNet headers:
  no

Public SDK dependency on ONNX Runtime headers:
  no

Release dependency on libwenet*.so:
  no

Release dependency on libonnxruntime.so/dll/dylib:
  yes, bundled and verified
```

In one sentence:

> **Vendor WeNet for maintainability, link only the needed WeNet runtime statically for reproducibility, and bundle ONNX Runtime dynamically for practical deployment.**
