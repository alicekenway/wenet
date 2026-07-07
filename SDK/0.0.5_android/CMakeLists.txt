cmake_minimum_required(VERSION 3.18)
project(wenet_static_ort_dynamic_sdk VERSION 0.0.5 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

include(cmake/Options.cmake)
include(cmake/OnnxRuntime.cmake)
include(cmake/WenetStaticRuntime.cmake)
include(cmake/SymbolVisibility.cmake)
include(cmake/KenLM.cmake)
include(cmake/FlashlightText.cmake)

find_package(Threads REQUIRED)

set(ASR_SDK_PLATFORM_LIBS Threads::Threads)
if(CMAKE_DL_LIBS)
  list(APPEND ASR_SDK_PLATFORM_LIBS ${CMAKE_DL_LIBS})
endif()
if(ANDROID)
  find_library(ASR_SDK_ANDROID_LOG_LIB log)
  if(ASR_SDK_ANDROID_LOG_LIB)
    list(APPEND ASR_SDK_PLATFORM_LIBS ${ASR_SDK_ANDROID_LOG_LIB})
  endif()
endif()

if(ASR_SDK_ENABLE_FLASHLIGHT_DECODER)
  set(ASR_SDK_FLASHLIGHT_DECODER_ENABLED 1)
else()
  set(ASR_SDK_FLASHLIGHT_DECODER_ENABLED 0)
endif()
if(ASR_SDK_ANDROID_REDUCED_ONNX)
  set(ASR_SDK_ANDROID_REDUCED_ONNX_ENABLED 1)
else()
  set(ASR_SDK_ANDROID_REDUCED_ONNX_ENABLED 0)
endif()
add_compile_definitions(
  ASR_SDK_ENABLE_FLASHLIGHT_DECODER=${ASR_SDK_FLASHLIGHT_DECODER_ENABLED}
  ASR_SDK_ANDROID_REDUCED_ONNX=${ASR_SDK_ANDROID_REDUCED_ONNX_ENABLED}
)

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
  src/sdk/c_api.cc
  src/sdk/result_json.cc
  src/sdk/status.cc
  src/sdk/version.cc
  src/utils/file_utils.cc
  src/utils/json.cc
  src/utils/timer.cc
)

if(ASR_SDK_ANDROID_REDUCED_ONNX)
  list(APPEND ASR_SDK_SOURCES
    src/android_reduced/greedy_asr_stream.cc
  )
else()
  list(APPEND ASR_SDK_SOURCES
    src/sdk/asr_stream.cc
    src/wenet_bridge/wenet_result_mapper.cc
    src/wenet_bridge/wenet_runtime_bridge.cc
    src/wenet_bridge/wenet_stream_adapter.cc
  )
endif()

set(SHERPA_ONNX_WENET_SOURCES
  src/sherpa_onnx_wenet/ctc_onnx_backend_factory.cc
  src/sherpa_onnx_wenet/ctc_greedy_decoder.cc
  src/sherpa_onnx_wenet/token_table.cc
  src/sherpa_onnx_wenet/wenet_ctc_onnx_backend.cc
  src/sherpa_onnx_wenet/whisper_feature_extractor.cc
  src/sherpa_onnx_wenet/zipformer2_ctc_onnx_backend.cc
)
if(ASR_SDK_ANDROID_REDUCED_ONNX)
  list(APPEND SHERPA_ONNX_WENET_SOURCES
    "${ASR_SDK_WENET_ONNX_ROOT}/frontend/fft.cc"
  )
endif()

set(ASR_SDK_FLASHLIGHT_DECODER_SOURCES)
if(ASR_SDK_ENABLE_FLASHLIGHT_DECODER)
  set(ASR_SDK_FLASHLIGHT_DECODER_SOURCES
    src/flashlight_decoder/decoded_hypothesis.cc
    src/flashlight_decoder/debug_trace.cc
    src/flashlight_decoder/flashlight_asr_stream.cc
    src/flashlight_decoder/flashlight_decoder_resource.cc
    src/flashlight_decoder/flashlight_ctc_stream_decoder.cc
    src/flashlight_decoder/flashlight_result_mapper.cc
    src/flashlight_decoder/lexicon_loader.cc
    src/flashlight_decoder/output_sequence_mapper.cc
    src/flashlight_decoder/word_dictionary.cc
  )
endif()

add_library(asr_sdk_ctc_core STATIC
  ${SHERPA_ONNX_WENET_SOURCES}
  ${ASR_SDK_FLASHLIGHT_DECODER_SOURCES}
)
add_library(asr_sdk::ctc_core ALIAS asr_sdk_ctc_core)
set_target_properties(asr_sdk_ctc_core PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
target_include_directories(asr_sdk_ctc_core
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE
    ${ASR_SDK_WENET_INCLUDE_DIRS}
)
target_link_libraries(asr_sdk_ctc_core
  PUBLIC
    asr_sdk::wenet_runtime_static
    onnxruntime
  PRIVATE
    ${ASR_SDK_PLATFORM_LIBS}
)
if(ASR_SDK_ENABLE_FLASHLIGHT_DECODER)
  target_link_libraries(asr_sdk_ctc_core PUBLIC asr_sdk::flashlight_text)
endif()
target_compile_definitions(asr_sdk_ctc_core PRIVATE
  ASR_SDK_VERSION_STRING="${PROJECT_VERSION}"
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
    asr_sdk::ctc_core
    onnxruntime
    ${ASR_SDK_PLATFORM_LIBS}
)

target_compile_definitions(asr_sdk PRIVATE
  ASR_SDK_VERSION_STRING="${PROJECT_VERSION}"
  ASR_SDK_ABI_VERSION=1
  ASR_SDK_ENABLE_FLASHLIGHT_DECODER=${ASR_SDK_FLASHLIGHT_DECODER_ENABLED}
  ASR_SDK_ANDROID_REDUCED_ONNX=${ASR_SDK_ANDROID_REDUCED_ONNX_ENABLED}
  ASR_SDK_WENET_COMMIT="${ASR_SDK_WENET_COMMIT}"
  ASR_SDK_ONNXRUNTIME_VERSION="${ASR_SDK_ONNXRUNTIME_VERSION}"
  ASR_SDK_FLASHLIGHT_TEXT_COMMIT="${ASR_SDK_FLASHLIGHT_TEXT_COMMIT}"
  ASR_SDK_KENLM_COMMIT="${ASR_SDK_KENLM_COMMIT}"
)

set_target_properties(asr_sdk PROPERTIES OUTPUT_NAME asr_sdk)
if(NOT ANDROID)
  set_target_properties(asr_sdk PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
  )
endif()

if(ASR_SDK_HIDE_INTERNAL_SYMBOLS)
  asr_sdk_apply_hidden_visibility(asr_sdk)
endif()

if(ASR_SDK_BUILD_TOOLS)
  add_executable(asr_stream_file
    cli/asr_stream_file.cc
    src/audio/wav_reader.cc
    src/utils/timer.cc
  )
  target_include_directories(asr_stream_file PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(asr_stream_file PRIVATE asr_sdk)
  set_target_properties(asr_stream_file PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN/../lib")

  add_executable(asr_batch_decode
    cli/asr_batch_decode.cc
    src/audio/wav_reader.cc
    src/utils/timer.cc
  )
  target_include_directories(asr_batch_decode PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(asr_batch_decode PRIVATE asr_sdk)
  set_target_properties(asr_batch_decode PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN/../lib")

  add_executable(asr_package_eval
    cli/asr_package_eval.cc
    src/audio/wav_reader.cc
    src/utils/json.cc
    src/utils/timer.cc
  )
  target_include_directories(asr_package_eval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(asr_package_eval PRIVATE asr_sdk asr_sdk::ctc_core)
  set_target_properties(asr_package_eval PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN/../lib")

  add_executable(inspect_package
    cli/inspect_package.cc
    src/package/model_package.cc
    src/package/model_package_validator.cc
    src/utils/file_utils.cc
    src/utils/json.cc
  )
  target_include_directories(inspect_package PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(inspect_package PRIVATE asr_sdk asr_sdk::ctc_core)
  set_target_properties(inspect_package PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN/../lib")

  add_executable(print_sdk_info cli/print_sdk_info.cc)
  target_link_libraries(print_sdk_info PRIVATE asr_sdk)
  set_target_properties(print_sdk_info PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN/../lib")

  if(ASR_SDK_ENABLE_FLASHLIGHT_DECODER)
    add_executable(zipformer_ctc_flashlight_main
      cli/zipformer_ctc_flashlight_main.cc
      src/audio/wav_reader.cc
      src/sdk/status.cc
      src/utils/timer.cc
    )
    target_include_directories(zipformer_ctc_flashlight_main PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${ASR_SDK_WENET_INCLUDE_DIRS}
    )
    target_link_libraries(zipformer_ctc_flashlight_main PRIVATE
      asr_sdk::ctc_core
      ${ASR_SDK_PLATFORM_LIBS}
    )
    set_target_properties(zipformer_ctc_flashlight_main PROPERTIES
      BUILD_RPATH "$ORIGIN"
      INSTALL_RPATH "$ORIGIN/../lib"
    )
  endif()

  if(ASR_SDK_ENABLE_LEGACY_WFST)
    add_executable(zipformer_ctc_wfst_main
      cli/zipformer_ctc_wfst_main.cc
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
      ${ASR_SDK_PLATFORM_LIBS}
    )
    set_target_properties(zipformer_ctc_wfst_main PROPERTIES
      BUILD_RPATH "$ORIGIN"
      INSTALL_RPATH "$ORIGIN/../lib"
    )
  endif()
endif()

if(ASR_SDK_BUILD_EXAMPLES)
  add_executable(simple_c_api examples/c/simple_c_api.c)
  target_link_libraries(simple_c_api PRIVATE asr_sdk)
  set_target_properties(simple_c_api PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN/../lib")

  add_executable(minimal_streaming examples/cpp/minimal_streaming.cc)
  target_link_libraries(minimal_streaming PRIVATE asr_sdk)
  set_target_properties(minimal_streaming PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN/../lib")
endif()

if(ASR_SDK_BUILD_TESTS AND ASR_SDK_ENABLE_FLASHLIGHT_DECODER)
  enable_testing()
  add_executable(output_sequence_mapper_test
    test/output_sequence_mapper_test.cc
  )
  target_include_directories(output_sequence_mapper_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(output_sequence_mapper_test PRIVATE asr_sdk::ctc_core)
  add_test(NAME output_sequence_mapper_test
           COMMAND output_sequence_mapper_test)

  add_executable(debug_trace_test
    test/debug_trace_test.cc
  )
  target_include_directories(debug_trace_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(debug_trace_test PRIVATE asr_sdk::ctc_core)
  add_test(NAME debug_trace_test
           COMMAND debug_trace_test)

  add_executable(model_package_decoder_options_test
    test/model_package_decoder_options_test.cc
    src/package/model_package.cc
    src/sdk/status.cc
    src/utils/file_utils.cc
    src/utils/json.cc
  )
  target_include_directories(model_package_decoder_options_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  add_test(NAME model_package_decoder_options_test
           COMMAND model_package_decoder_options_test)
endif()

include(cmake/InstallRules.cmake)
