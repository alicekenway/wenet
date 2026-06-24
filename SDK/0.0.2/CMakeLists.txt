cmake_minimum_required(VERSION 3.18)
project(wenet_static_ort_dynamic_sdk VERSION 0.0.2 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

include(cmake/Options.cmake)
include(cmake/OnnxRuntimeDynamic.cmake)
include(cmake/WenetStaticRuntime.cmake)
include(cmake/SymbolVisibility.cmake)

set(ASR_SDK_PUBLIC_HEADERS
  include/asr_sdk/asr_engine.h
  include/asr_sdk/c_api.h
  include/asr_sdk/config.h
  include/asr_sdk/export.h
  include/asr_sdk/result.h
  include/asr_sdk/status.h
  include/asr_sdk/stream.h
  include/asr_sdk/version.h
)

set(ASR_SDK_SOURCES
  src/audio/wav_reader.cc
  src/package/model_package.cc
  src/package/model_package_validator.cc
  src/sdk/asr_engine.cc
  src/sdk/asr_stream.cc
  src/sdk/c_api.cc
  src/sdk/result_json.cc
  src/sdk/status.cc
  src/sdk/version.cc
  src/utils/file_utils.cc
  src/utils/json.cc
  src/utils/timer.cc
  src/wenet_bridge/wenet_result_mapper.cc
  src/wenet_bridge/wenet_runtime_bridge.cc
  src/wenet_bridge/wenet_stream_adapter.cc
)

set(SHERPA_ONNX_WENET_SOURCES
  src/sherpa_onnx_wenet/ctc_greedy_decoder.cc
  src/sherpa_onnx_wenet/token_table.cc
  src/sherpa_onnx_wenet/whisper_feature_extractor.cc
  src/sherpa_onnx_wenet/zipformer2_ctc_onnx_backend.cc
)

add_library(asr_sdk SHARED
  ${ASR_SDK_PUBLIC_HEADERS}
  ${ASR_SDK_SOURCES}
)
add_library(asr_sdk::asr_sdk ALIAS asr_sdk)

target_include_directories(asr_sdk
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ASR_SDK_WENET_INCLUDE_DIRS}
)

target_link_libraries(asr_sdk
  PRIVATE
    asr_sdk::wenet_runtime_static
    onnxruntime
    dl
    pthread
)

target_compile_definitions(asr_sdk PRIVATE
  ASR_SDK_VERSION_STRING="${PROJECT_VERSION}"
  ASR_SDK_ABI_VERSION=1
  ASR_SDK_WENET_COMMIT="${ASR_SDK_WENET_COMMIT}"
  ASR_SDK_ONNXRUNTIME_VERSION="${ASR_SDK_ONNXRUNTIME_VERSION}"
)

set_target_properties(asr_sdk PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
  OUTPUT_NAME asr_sdk
)

if(ASR_SDK_HIDE_INTERNAL_SYMBOLS)
  asr_sdk_apply_hidden_visibility(asr_sdk)
endif()

if(ASR_SDK_BUILD_TOOLS)
  add_executable(asr_stream_file
    tools/asr_stream_file.cc
    src/audio/wav_reader.cc
    src/utils/timer.cc
  )
  target_include_directories(asr_stream_file PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(asr_stream_file PRIVATE asr_sdk)
  set_target_properties(asr_stream_file PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN/../lib")

  add_executable(asr_batch_decode
    tools/asr_batch_decode.cc
    src/audio/wav_reader.cc
    src/utils/timer.cc
  )
  target_include_directories(asr_batch_decode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(asr_batch_decode PRIVATE asr_sdk)
  set_target_properties(asr_batch_decode PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN/../lib")

  add_executable(inspect_package
    tools/inspect_package.cc
    src/package/model_package.cc
    src/package/model_package_validator.cc
    src/utils/file_utils.cc
    src/utils/json.cc
  )
  target_include_directories(inspect_package PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(inspect_package PRIVATE asr_sdk)
  set_target_properties(inspect_package PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN/../lib")

  add_executable(print_build_info tools/print_build_info.cc)
  target_link_libraries(print_build_info PRIVATE asr_sdk)
  set_target_properties(print_build_info PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN/../lib")

  add_executable(zipformer_ctc_wfst_main
    tools/zipformer_ctc_wfst_main.cc
    ${SHERPA_ONNX_WENET_SOURCES}
    src/audio/wav_reader.cc
    src/sdk/status.cc
    src/utils/timer.cc
  )
  target_include_directories(zipformer_ctc_wfst_main PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ASR_SDK_WENET_INCLUDE_DIRS}
  )
  target_link_libraries(zipformer_ctc_wfst_main PRIVATE
    ${ASR_SDK_WENET_STATIC_LIBS}
    "${ASR_SDK_WENET_BUILD_DIR}/decoder/libdecoder.a"
    onnxruntime
    dl
    pthread
  )
  set_target_properties(zipformer_ctc_wfst_main PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN/../lib"
  )
endif()

if(ASR_SDK_BUILD_EXAMPLES)
  add_executable(simple_c_api examples/c/simple_c_api.c)
  target_link_libraries(simple_c_api PRIVATE asr_sdk)
  set_target_properties(simple_c_api PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN/../lib")

  add_executable(minimal_streaming examples/cpp/minimal_streaming.cc)
  target_link_libraries(minimal_streaming PRIVATE asr_sdk)
  set_target_properties(minimal_streaming PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN/../lib")
endif()

include(cmake/InstallRules.cmake)
