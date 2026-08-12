option(ASR_SDK_STATIC_WENET "Link WeNet runtime statically into SDK" ON)
option(ASR_SDK_DYNAMIC_ONNXRUNTIME "Use dynamic ONNX Runtime" ON)
option(ASR_SDK_BUILD_TOOLS "Build SDK CLI tools" ON)
option(ASR_SDK_BUILD_EXAMPLES "Build SDK examples" ON)
option(ASR_SDK_BUILD_TESTS "Build SDK tests" ON)
option(ASR_SDK_HIDE_INTERNAL_SYMBOLS "Hide internal symbols" ON)

set(ASR_SDK_REQUIRED_ORT_VERSION
    "1.25.1"
    CACHE STRING "Required ONNX Runtime version")
set(ASR_SDK_ONNXRUNTIME_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime"
    CACHE PATH "Pinned ONNX Runtime root containing include/ and lib/")
option(ASR_SDK_ALLOW_ORT_VERSION_MISMATCH
       "Allow local experimentation with a non-required ONNX Runtime version"
       OFF)

option(ASR_SDK_ENABLE_FLASHLIGHT_DECODER
       "Build Flashlight lexicon + KenLM decoder" ON)
option(ASR_SDK_ENABLE_LEGACY_WFST
       "Build legacy WFST comparison decoder" OFF)
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
    "/home/jinyang_wang/Dev/ASR/ASR_wenet/wenet"
    CACHE PATH "Path to the pinned alicekenway/wenet checkout")
set(ASR_SDK_WENET_BUILD_DIR
    "${ASR_SDK_WENET_ROOT}/runtime/onnxruntime/build"
    CACHE PATH "Path to an existing WeNet ONNX runtime build directory")
