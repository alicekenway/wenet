set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

include(cmake/Options.cmake)
if(NOT ASR_SDK_REQUIRED_ORT_VERSION STREQUAL "1.16.3")
  message(FATAL_ERROR "ASR SDK 0.0.15 release builds require ONNX Runtime 1.16.3")
endif()
include(cmake/OnnxRuntime.cmake)
include(cmake/WeTextProcessing.cmake)
include(cmake/WenetStaticRuntime.cmake)
include(cmake/SymbolVisibility.cmake)
include(cmake/KenLM.cmake)
include(cmake/FlashlightText.cmake)

set(ASR_SDK_SOURCES ${ASR_SDK_BASE_SOURCES} ${ASR_SDK_LEGACY_SOURCES})
set(SHERPA_ONNX_WENET_SOURCES ${ASR_SDK_CTC_SOURCES})

set(ASR_SDK_FLASHLIGHT_DECODER_SOURCES)
if(ASR_SDK_ENABLE_FLASHLIGHT_DECODER)
  set(ASR_SDK_FLASHLIGHT_DECODER_SOURCES ${ASR_SDK_FLASHLIGHT_SOURCES})
endif()

add_library(asr_sdk_ctc_core STATIC
  ${SHERPA_ONNX_WENET_SOURCES}
  ${ASR_SDK_FLASHLIGHT_DECODER_SOURCES}
)
add_library(asr_sdk::ctc_core ALIAS asr_sdk_ctc_core)
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
    dl
    pthread
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
    ${ASR_SDK_WETEXT_ROOT}/runtime
    ${ASR_SDK_WENET_INCLUDE_DIRS}
)

target_link_libraries(asr_sdk
  PRIVATE
    asr_sdk::wenet_runtime_static
    asr_sdk::ctc_core
    onnxruntime
    dl
    pthread
)

target_compile_definitions(asr_sdk PRIVATE
  ASR_SDK_PRODUCTION_FLASHLIGHT_ONLY=0
  ASR_SDK_VERSION_STRING="${PROJECT_VERSION}"
  ASR_SDK_ABI_VERSION=5
  ASR_SDK_WENET_COMMIT="${ASR_SDK_WENET_COMMIT}"
  ASR_SDK_ONNXRUNTIME_VERSION="${ASR_SDK_ONNXRUNTIME_VERSION}"
  ASR_SDK_FLASHLIGHT_TEXT_COMMIT="${ASR_SDK_FLASHLIGHT_TEXT_COMMIT}"
  ASR_SDK_KENLM_COMMIT="${ASR_SDK_KENLM_COMMIT}"
  ASR_SDK_WETEXT_COMMIT="${ASR_SDK_WETEXT_COMMIT}"
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
    src/itn/itn_processor.cc
    src/utils/json.cc
    src/utils/timer.cc
  )
  target_include_directories(asr_package_eval PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
  target_link_libraries(asr_package_eval PRIVATE
    asr_sdk asr_sdk::ctc_core asr_sdk_wetext_processor)
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

  add_executable(print_build_info cli/print_sdk_info.cc)
  target_link_libraries(print_build_info PRIVATE asr_sdk)
  set_target_properties(print_build_info PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN/../lib")

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
      dl
      pthread
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
      dl
      pthread
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

  add_executable(pre_lm_mapping_test
    test/pre_lm_mapping_test.cc
    src/sdk/status.cc
  )
  target_include_directories(pre_lm_mapping_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(pre_lm_mapping_test PRIVATE asr_sdk::ctc_core)
  add_test(NAME pre_lm_mapping_test
           COMMAND pre_lm_mapping_test)

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

  add_executable(contact_class_lm_test
    test/contact_class_lm_test.cc
    src/sdk/status.cc
  )
  target_include_directories(contact_class_lm_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(contact_class_lm_test PRIVATE asr_sdk::ctc_core)
  add_test(NAME contact_class_lm_test
           COMMAND contact_class_lm_test)

  add_executable(multi_trie_lexicon_decoder_test
    test/multi_trie_lexicon_decoder_test.cc
  )
  target_include_directories(multi_trie_lexicon_decoder_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
  )
  target_link_libraries(multi_trie_lexicon_decoder_test PRIVATE
    asr_sdk::ctc_core)
  add_test(NAME multi_trie_lexicon_decoder_test
           COMMAND multi_trie_lexicon_decoder_test)

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

  add_executable(itn_processor_test
    test/itn_processor_test.cc
    src/itn/itn_processor.cc
    src/sdk/status.cc
  )
  target_include_directories(itn_processor_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${ASR_SDK_WETEXT_ROOT}/runtime
  )
  target_link_libraries(itn_processor_test PRIVATE
    asr_sdk_wetext_processor Threads::Threads)
  add_test(NAME itn_processor_test
    COMMAND itn_processor_test
      ${CMAKE_CURRENT_SOURCE_DIR}/../../ITN/english/export/en_itn_tagger.fst
      ${CMAKE_CURRENT_SOURCE_DIR}/../../ITN/english/export/en_itn_verbalizer.fst)

  find_program(ASR_SDK_BASH_EXECUTABLE bash)
  if(ASR_SDK_BASH_EXECUTABLE)
    add_test(NAME generate_contact_bias_arpa_test
             COMMAND ${ASR_SDK_BASH_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/test/generate_contact_bias_arpa_test.sh)
    add_test(NAME prepare_flashlight_runtime_package_test
             COMMAND ${ASR_SDK_BASH_EXECUTABLE}
                     ${CMAKE_CURRENT_SOURCE_DIR}/test/prepare_flashlight_runtime_package_test.sh)
  endif()
endif()

include(cmake/InstallRules.cmake)
