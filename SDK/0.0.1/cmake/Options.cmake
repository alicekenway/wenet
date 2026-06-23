option(ASR_SDK_STATIC_WENET "Link WeNet runtime statically into SDK" ON)
option(ASR_SDK_DYNAMIC_ONNXRUNTIME "Use dynamic ONNX Runtime" ON)
option(ASR_SDK_BUILD_TOOLS "Build SDK CLI tools" ON)
option(ASR_SDK_BUILD_EXAMPLES "Build SDK examples" ON)
option(ASR_SDK_HIDE_INTERNAL_SYMBOLS "Hide internal symbols" ON)

set(ASR_SDK_WENET_ROOT
    "/home/jinyang_wang/Dev/ASR/ASR_wenet/wenet"
    CACHE PATH "Path to the pinned alicekenway/wenet checkout")
set(ASR_SDK_WENET_BUILD_DIR
    "${ASR_SDK_WENET_ROOT}/runtime/onnxruntime/build"
    CACHE PATH "Path to an existing WeNet ONNX runtime build directory")
set(ASR_SDK_ONNXRUNTIME_LIB
    "/home/jinyang_wang/miniforge3/envs/wenet/lib/python3.10/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.2"
    CACHE FILEPATH "Pinned ONNX Runtime shared library")
set(ASR_SDK_ONNXRUNTIME_INCLUDE_DIR
    "${ASR_SDK_WENET_ROOT}/runtime/onnxruntime/fc_base/onnxruntime-src/include"
    CACHE PATH "ONNX Runtime headers used to compile the pinned WeNet runtime")
