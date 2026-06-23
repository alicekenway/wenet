cmake_minimum_required(VERSION 3.18)

project(wenet_lite_sdk
  VERSION 0.1.0
  LANGUAGES C CXX)

option(WENETSDK_BUILD_TOOLS "Build CLI tools" ON)
option(WENETSDK_BUILD_TESTS "Build tests" ON)
option(WENETSDK_BUILD_EXAMPLES "Build examples" ON)
option(WENETSDK_ENABLE_ONNX "Enable ONNX Runtime backend" OFF)
option(WENETSDK_ENABLE_OPENFST "Enable OpenFst WFST decoder backend" OFF)
option(WENETSDK_ENABLE_PORTAUDIO "Enable PortAudio microphone demo" OFF)
option(WENETSDK_ENABLE_SANITIZERS "Enable address/undefined sanitizers" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(WENETSDK_ENABLE_SANITIZERS)
  include(Sanitizers)
endif()

if(WENETSDK_ENABLE_ONNX)
  find_package(ONNXRuntime REQUIRED)
  get_filename_component(WENETSDK_ONNXRUNTIME_LIBRARY_DIR
    "${ONNXRuntime_LIBRARY}" DIRECTORY)
  set(WENETSDK_ONNXRUNTIME_RUNTIME_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime_runtime")
  file(MAKE_DIRECTORY "${WENETSDK_ONNXRUNTIME_RUNTIME_DIR}")
  file(CREATE_LINK
    "${ONNXRuntime_LIBRARY}"
    "${WENETSDK_ONNXRUNTIME_RUNTIME_DIR}/libonnxruntime.so.1"
    SYMBOLIC COPY_ON_ERROR)
endif()

if(WENETSDK_ENABLE_OPENFST)
  find_package(OpenFst REQUIRED)
endif()

if(WENETSDK_ENABLE_PORTAUDIO)
  find_package(PortAudio REQUIRED)
endif()

add_library(wenet_sdk
  src/core/asr_engine.cc
  src/core/c_api.cc
  src/core/recognizer.cc
  src/core/result_builder.cc
  src/core/stream_session.cc
  src/decoder/blank_skipper.cc
  src/decoder/ctc_prefix_decoder.cc
  src/decoder/ctc_wfst_decoder.cc
  src/decoder/fst_loader.cc
  src/decoder/greedy_ctc_decoder.cc
  src/decoder/nbest.cc
  src/decoder/symbol_table.cc
  src/decoder/token.cc
  src/frontend/cmvn.cc
  src/frontend/fbank.cc
  src/frontend/feature_pipeline.cc
  src/frontend/resampler.cc
  src/frontend/ring_buffer.cc
  src/frontend/window.cc
  src/io/wav_reader.cc
  src/io/wav_writer.cc
  src/model/model_metadata.cc
  src/model/onnx_ctc_model.cc
  src/model/onnx_session_pool.cc
  src/model/tensor_utils.cc
  src/postprocess/endpoint.cc
  src/postprocess/text_normalizer.cc
  src/postprocess/timestamp_estimator.cc
  src/utils/checksum.cc
  src/utils/json.cc
  src/utils/logging.cc
  src/utils/status.cc
  src/utils/thread_pool.cc
  src/utils/timer.cc)

target_include_directories(wenet_sdk
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_compile_definitions(wenet_sdk
  PUBLIC
    WENETSDK_VERSION_MAJOR=${PROJECT_VERSION_MAJOR}
    WENETSDK_VERSION_MINOR=${PROJECT_VERSION_MINOR}
    WENETSDK_VERSION_PATCH=${PROJECT_VERSION_PATCH})

if(WENETSDK_ENABLE_ONNX)
  target_compile_definitions(wenet_sdk PRIVATE WENETSDK_ENABLE_ONNX=1)
  target_link_libraries(wenet_sdk PRIVATE ONNXRuntime::ONNXRuntime)
endif()

if(WENETSDK_ENABLE_OPENFST)
  target_compile_definitions(wenet_sdk PRIVATE WENETSDK_ENABLE_OPENFST=1)
  target_link_libraries(wenet_sdk PRIVATE OpenFst::fst)
endif()

if(MSVC)
  target_compile_options(wenet_sdk PRIVATE /W4)
else()
  target_compile_options(wenet_sdk PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(WENETSDK_BUILD_TOOLS)
  foreach(tool asr_stream_file asr_stream_mic benchmark inspect_fst inspect_model validate_model_package)
    add_executable(${tool} tools/${tool}.cc)
    target_include_directories(${tool} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(${tool} PRIVATE
      $<$<BOOL:${WENETSDK_ENABLE_ONNX}>:WENETSDK_ENABLE_ONNX=1>
      $<$<BOOL:${WENETSDK_ENABLE_OPENFST}>:WENETSDK_ENABLE_OPENFST=1>
      $<$<BOOL:${WENETSDK_ENABLE_PORTAUDIO}>:WENETSDK_ENABLE_PORTAUDIO=1>)
    if(WENETSDK_ENABLE_ONNX)
      set_target_properties(${tool} PROPERTIES
        BUILD_RPATH "${WENETSDK_ONNXRUNTIME_RUNTIME_DIR}")
    endif()
    target_link_libraries(${tool} PRIVATE wenet_sdk)
    if(WENETSDK_ENABLE_ONNX AND tool STREQUAL "inspect_model")
      target_link_libraries(${tool} PRIVATE ONNXRuntime::ONNXRuntime)
    endif()
    if(WENETSDK_ENABLE_PORTAUDIO AND tool STREQUAL "asr_stream_mic")
      target_link_libraries(${tool} PRIVATE PortAudio::PortAudio)
    endif()
  endforeach()
endif()

if(WENETSDK_BUILD_EXAMPLES)
  add_executable(minimal_streaming examples/cpp/minimal_streaming.cc)
  if(WENETSDK_ENABLE_ONNX)
    set_target_properties(minimal_streaming PROPERTIES
      BUILD_RPATH "${WENETSDK_ONNXRUNTIME_RUNTIME_DIR}")
  endif()
  target_link_libraries(minimal_streaming PRIVATE wenet_sdk)

  add_executable(batch_files examples/cpp/batch_files.cc)
  target_include_directories(batch_files PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  if(WENETSDK_ENABLE_ONNX)
    set_target_properties(batch_files PROPERTIES
      BUILD_RPATH "${WENETSDK_ONNXRUNTIME_RUNTIME_DIR}")
  endif()
  target_link_libraries(batch_files PRIVATE wenet_sdk)

  add_executable(pseudo_audio_callback
    examples/embedded/pseudo_audio_callback.cc)
  if(WENETSDK_ENABLE_ONNX)
    set_target_properties(pseudo_audio_callback PROPERTIES
      BUILD_RPATH "${WENETSDK_ONNXRUNTIME_RUNTIME_DIR}")
  endif()
  target_link_libraries(pseudo_audio_callback PRIVATE wenet_sdk)

  add_executable(simple_c_api examples/c/simple_c_api.c)
  if(WENETSDK_ENABLE_ONNX)
    set_target_properties(simple_c_api PROPERTIES
      BUILD_RPATH "${WENETSDK_ONNXRUNTIME_RUNTIME_DIR}")
  endif()
  target_link_libraries(simple_c_api PRIVATE wenet_sdk)
endif()

if(WENETSDK_BUILD_TESTS)
  enable_testing()

  foreach(test_name
      test_blank_skipper
      test_checksum
      test_cmvn
      test_ctc_collapse
      test_ctc_prefix_decoder
      test_ctc_wfst_decoder
      test_endpoint
      test_feature_pipeline
      test_model_package
      test_onnx_ctc_model
      test_streaming_boundaries
      test_symbol_table)
    add_executable(${test_name} tests/unit/${test_name}.cc)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(${test_name} PRIVATE
      $<$<BOOL:${WENETSDK_ENABLE_ONNX}>:WENETSDK_ENABLE_ONNX=1>
      $<$<BOOL:${WENETSDK_ENABLE_OPENFST}>:WENETSDK_ENABLE_OPENFST=1>)
    if(WENETSDK_ENABLE_ONNX)
      set_target_properties(${test_name} PROPERTIES
        BUILD_RPATH "${WENETSDK_ONNXRUNTIME_RUNTIME_DIR}")
    endif()
    target_link_libraries(${test_name} PRIVATE wenet_sdk)
    if(WENETSDK_ENABLE_OPENFST AND test_name STREQUAL "test_ctc_wfst_decoder")
      target_link_libraries(${test_name} PRIVATE OpenFst::fst)
    endif()
    add_test(NAME ${test_name} COMMAND ${test_name})
    set_tests_properties(${test_name} PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    if(WENETSDK_ENABLE_ONNX)
      set_tests_properties(${test_name} PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${WENETSDK_ONNXRUNTIME_RUNTIME_DIR}:$ENV{LD_LIBRARY_PATH}")
    endif()
  endforeach()

  if(NOT WENETSDK_ENABLE_OPENFST)
    set_tests_properties(test_ctc_wfst_decoder PROPERTIES DISABLED TRUE)
  endif()
  if(NOT WENETSDK_ENABLE_ONNX)
    set_tests_properties(test_onnx_ctc_model PROPERTIES DISABLED TRUE)
  endif()

  foreach(integration_test
      test_decode_toy_fst
      test_decode_wav
      test_model_forward)
    add_executable(${integration_test} tests/integration/${integration_test}.cc)
    target_include_directories(${integration_test} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
    target_compile_definitions(${integration_test} PRIVATE
      $<$<BOOL:${WENETSDK_ENABLE_ONNX}>:WENETSDK_ENABLE_ONNX=1>
      $<$<BOOL:${WENETSDK_ENABLE_OPENFST}>:WENETSDK_ENABLE_OPENFST=1>)
    if(WENETSDK_ENABLE_ONNX)
      set_target_properties(${integration_test} PROPERTIES
        BUILD_RPATH "${WENETSDK_ONNXRUNTIME_RUNTIME_DIR}")
    endif()
    target_link_libraries(${integration_test} PRIVATE wenet_sdk)
    if(WENETSDK_ENABLE_OPENFST AND integration_test STREQUAL "test_decode_toy_fst")
      target_link_libraries(${integration_test} PRIVATE OpenFst::fst)
    endif()
    add_test(NAME ${integration_test} COMMAND ${integration_test})
    set_tests_properties(${integration_test} PROPERTIES
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    if(WENETSDK_ENABLE_ONNX)
      set_tests_properties(${integration_test} PROPERTIES
        ENVIRONMENT "LD_LIBRARY_PATH=${WENETSDK_ONNXRUNTIME_RUNTIME_DIR}:$ENV{LD_LIBRARY_PATH}")
    endif()
  endforeach()

  if(NOT WENETSDK_ENABLE_OPENFST)
    set_tests_properties(test_decode_toy_fst PROPERTIES DISABLED TRUE)
  endif()
endif()

install(TARGETS wenet_sdk
  EXPORT wenet_sdkTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin)

install(DIRECTORY include/wenet_sdk DESTINATION include)
