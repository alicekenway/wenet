set(ASR_SDK_WETEXT_ROOT
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/WeTextProcessing")
set(ASR_SDK_WETEXT_COMMIT "57f85850156c519690f0e60b2096f2a21aa7fa4c")

add_library(asr_sdk_wetext_utils STATIC
  "${ASR_SDK_WETEXT_ROOT}/runtime/utils/wetext_string.cc")
target_include_directories(asr_sdk_wetext_utils PUBLIC
  "${ASR_SDK_WETEXT_ROOT}/runtime"
  "${ASR_SDK_WENET_ROOT}/runtime/onnxruntime/fc_base/openfst-src/src/include"
  "${ASR_SDK_WENET_ROOT}/runtime/onnxruntime/fc_base/gflags-build/include"
  "${ASR_SDK_WENET_ROOT}/runtime/onnxruntime/fc_base/glog-src/src"
  "${ASR_SDK_WENET_ROOT}/runtime/onnxruntime/fc_base/glog-build")
target_link_libraries(asr_sdk_wetext_utils PUBLIC
  "${ASR_SDK_WENET_ROOT}/runtime/onnxruntime/fc_base/glog-build/libglog.a"
  "${ASR_SDK_WENET_ROOT}/runtime/onnxruntime/fc_base/gflags-build/libgflags_nothreads.a")
set_target_properties(asr_sdk_wetext_utils PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(asr_sdk_wetext_processor STATIC
  "${ASR_SDK_WETEXT_ROOT}/runtime/processor/wetext_processor.cc"
  "${ASR_SDK_WETEXT_ROOT}/runtime/processor/wetext_token_parser.cc")
target_include_directories(asr_sdk_wetext_processor PUBLIC
  "${ASR_SDK_WETEXT_ROOT}/runtime"
  "${ASR_SDK_WENET_ROOT}/runtime/onnxruntime/fc_base/openfst-src/src/include")
target_link_libraries(asr_sdk_wetext_processor PUBLIC
  "${ASR_SDK_WENET_ROOT}/runtime/onnxruntime/fc_base/openfst-build/src/lib/libfst.a"
  asr_sdk_wetext_utils
  dl)
set_target_properties(asr_sdk_wetext_processor PROPERTIES POSITION_INDEPENDENT_CODE ON)
