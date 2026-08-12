if(NOT ASR_SDK_STATIC_WENET)
  message(FATAL_ERROR "ASR_SDK_STATIC_WENET=OFF is not supported by this SDK plan")
endif()

if(NOT EXISTS "${ASR_SDK_WENET_ROOT}/runtime/core/api/wenet_api.cc")
  message(FATAL_ERROR "WeNet checkout missing or invalid: ${ASR_SDK_WENET_ROOT}")
endif()

if(NOT EXISTS "${ASR_SDK_WENET_BUILD_DIR}/decoder/libdecoder.a")
  message(FATAL_ERROR
    "WeNet ONNX runtime static archives not found in ${ASR_SDK_WENET_BUILD_DIR}. "
    "Build the forked WeNet runtime/onnxruntime tree first.")
endif()

execute_process(
  COMMAND git -C "${ASR_SDK_WENET_ROOT}" rev-parse HEAD
  OUTPUT_VARIABLE ASR_SDK_WENET_COMMIT
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)
if(NOT ASR_SDK_WENET_COMMIT)
  set(ASR_SDK_WENET_COMMIT "unknown")
endif()

set(ASR_SDK_WENET_ONNX_ROOT "${ASR_SDK_WENET_ROOT}/runtime/onnxruntime")
set(ASR_SDK_WENET_FC_BASE "${ASR_SDK_WENET_ONNX_ROOT}/fc_base")

set(ASR_SDK_WENET_INCLUDE_DIRS
  "${ASR_SDK_ONNXRUNTIME_INCLUDE_DIR}"
  "${ASR_SDK_WENET_FC_BASE}/gflags-build/include"
  "${ASR_SDK_WENET_FC_BASE}/glog-src/src"
  "${ASR_SDK_WENET_FC_BASE}/glog-build"
  "${ASR_SDK_WENET_FC_BASE}/openfst-src/src/include"
  "${ASR_SDK_WENET_ONNX_ROOT}"
  "${ASR_SDK_WENET_ONNX_ROOT}/kaldi"
  "${ASR_SDK_WENET_FC_BASE}/wetextprocessing-src/runtime"
)

set(ASR_SDK_WENET_STATIC_LIBS
  "${ASR_SDK_WENET_BUILD_DIR}/decoder/libdecoder.a"
  "${ASR_SDK_WENET_BUILD_DIR}/kaldi/libkaldi-decoder.a"
  "${ASR_SDK_WENET_BUILD_DIR}/kaldi/libkaldi-util.a"
  "${ASR_SDK_WENET_BUILD_DIR}/frontend/libfrontend.a"
  "${ASR_SDK_WENET_BUILD_DIR}/post_processor/libpost_processor.a"
  "${ASR_SDK_WENET_BUILD_DIR}/fc_base/wetextprocessing-src/runtime/processor/libwetext_processor.a"
  "${ASR_SDK_WENET_BUILD_DIR}/fc_base/wetextprocessing-src/runtime/utils/libwetext_utils.a"
  "${ASR_SDK_WENET_BUILD_DIR}/utils/libutils.a"
  "${ASR_SDK_WENET_FC_BASE}/openfst-build/src/lib/libfst.a"
  "${ASR_SDK_WENET_FC_BASE}/glog-build/libglog.a"
  "${ASR_SDK_WENET_FC_BASE}/gflags-build/libgflags_nothreads.a"
)

foreach(lib IN LISTS ASR_SDK_WENET_STATIC_LIBS)
  if(NOT EXISTS "${lib}")
    message(FATAL_ERROR "Required WeNet static dependency not found: ${lib}")
  endif()
endforeach()

add_library(asr_sdk_wenet_runtime_static INTERFACE)
target_link_libraries(asr_sdk_wenet_runtime_static INTERFACE
  ${ASR_SDK_WENET_STATIC_LIBS}
)
add_library(asr_sdk::wenet_runtime_static ALIAS asr_sdk_wenet_runtime_static)

message(STATUS "ASR SDK WeNet root: ${ASR_SDK_WENET_ROOT}")
message(STATUS "ASR SDK WeNet commit: ${ASR_SDK_WENET_COMMIT}")
message(STATUS "ASR SDK WeNet linkage: static/private")
