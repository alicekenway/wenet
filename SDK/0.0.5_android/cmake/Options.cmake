option(ASR_SDK_STATIC_WENET "Link WeNet runtime statically into SDK" ON)
option(ASR_SDK_DYNAMIC_ONNXRUNTIME "Use dynamic ONNX Runtime" ON)

if(ANDROID)
  option(ASR_SDK_BUILD_TOOLS "Build SDK CLI tools" OFF)
  option(ASR_SDK_BUILD_EXAMPLES "Build SDK examples" OFF)
  option(ASR_SDK_BUILD_TESTS "Build SDK tests" OFF)
else()
  option(ASR_SDK_BUILD_TOOLS "Build SDK CLI tools" ON)
  option(ASR_SDK_BUILD_EXAMPLES "Build SDK examples" ON)
  option(ASR_SDK_BUILD_TESTS "Build SDK tests" ON)
endif()

option(ASR_SDK_HIDE_INTERNAL_SYMBOLS "Hide internal symbols" ON)

if(ANDROID)
  option(ASR_SDK_ANDROID_REDUCED_ONNX
         "Build Android with ONNX CTC greedy backend and no WeNet static runtime"
         ON)
else()
  option(ASR_SDK_ANDROID_REDUCED_ONNX
         "Build Android with ONNX CTC greedy backend and no WeNet static runtime"
         OFF)
endif()

set(ASR_SDK_REQUIRED_ORT_VERSION
    "1.25.1"
    CACHE STRING "Required ONNX Runtime version")
set(ASR_SDK_ONNXRUNTIME_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime"
    CACHE PATH "Pinned ONNX Runtime root containing include/ and lib/")
option(ASR_SDK_ALLOW_ORT_VERSION_MISMATCH
       "Allow local experimentation with a non-required ONNX Runtime version"
       OFF)

if(ANDROID)
  option(ASR_SDK_ENABLE_FLASHLIGHT_DECODER
         "Build Flashlight lexicon + KenLM decoder" OFF)
  option(ASR_SDK_ENABLE_LEGACY_WFST
         "Build legacy WFST comparison decoder" OFF)
else()
  option(ASR_SDK_ENABLE_FLASHLIGHT_DECODER
         "Build Flashlight lexicon + KenLM decoder" ON)
  option(ASR_SDK_ENABLE_LEGACY_WFST
         "Build legacy WFST comparison decoder" OFF)
endif()

set(ASR_SDK_FLASHLIGHT_TEXT_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/flashlight-text"
    CACHE PATH "Pinned Flashlight-Text source")
set(ASR_SDK_KENLM_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/kenlm"
    CACHE PATH "Pinned KenLM source")
set(ASR_SDK_KENLM_INSTALL_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/kenlm-install"
    CACHE PATH "Local KenLM install containing include/, lib/, and bin/")

set(ASR_SDK_WENET_ROOT
    ""
    CACHE PATH "Path to the pinned alicekenway/wenet checkout")
set(ASR_SDK_WENET_BUILD_DIR
    ""
    CACHE PATH "Path to an existing WeNet ONNX runtime build directory")

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
