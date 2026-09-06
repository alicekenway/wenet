set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_NO_PACKAGE_REGISTRY ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")

set(ASR_SDK_ONNXRUNTIME_ROOT "" CACHE PATH "Extracted ONNX Runtime Android AAR")
set(ASR_SDK_WENET_ROOT "" CACHE PATH "Pinned WeNet checkout")
set(ASR_SDK_REQUIRED_ORT_VERSION "1.16.3" CACHE STRING "Required ONNX Runtime version")
option(ASR_SDK_BUILD_TOOLS "Build native Android test tools" ON)
option(ASR_SDK_HIDE_INTERNAL_SYMBOLS "Hide internal symbols" ON)

if(NOT ASR_SDK_REQUIRED_ORT_VERSION STREQUAL "1.16.3")
  message(FATAL_ERROR "The Android ARM64 release is pinned to ONNX Runtime 1.16.3")
endif()
if(NOT EXISTS "${ASR_SDK_WENET_ROOT}/runtime/onnxruntime/frontend/fbank.h")
  message(FATAL_ERROR "ASR_SDK_WENET_ROOT is missing the WeNet frontend")
endif()

set(ASR_SDK_ONNXRUNTIME_INCLUDE_DIR "${ASR_SDK_ONNXRUNTIME_ROOT}/headers")
set(ASR_SDK_ONNXRUNTIME_LIB
    "${ASR_SDK_ONNXRUNTIME_ROOT}/jni/${ANDROID_ABI}/libonnxruntime.so")
if(NOT EXISTS "${ASR_SDK_ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h" OR
   NOT EXISTS "${ASR_SDK_ONNXRUNTIME_LIB}")
  message(FATAL_ERROR
    "ASR_SDK_ONNXRUNTIME_ROOT must be an extracted ONNX Runtime Android AAR")
endif()
file(STRINGS "${ASR_SDK_ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h"
     _ort_api_line REGEX "^#define ORT_API_VERSION [0-9]+$")
if(NOT _ort_api_line STREQUAL "#define ORT_API_VERSION 16")
  message(FATAL_ERROR "ONNX Runtime 1.16.3 headers must expose API version 16")
endif()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ASR_SDK_ONNXRUNTIME_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${ASR_SDK_ONNXRUNTIME_INCLUDE_DIR}")

find_package(Threads REQUIRED)
set(ASR_SDK_PLATFORM_LIBS Threads::Threads)
if(CMAKE_DL_LIBS)
  list(APPEND ASR_SDK_PLATFORM_LIBS ${CMAKE_DL_LIBS})
endif()
find_library(ASR_SDK_ANDROID_LOG_LIB log REQUIRED)
list(APPEND ASR_SDK_PLATFORM_LIBS ${ASR_SDK_ANDROID_LOG_LIB})

# Current feature extraction and ITN use the sources already pinned by WeNet.
set(_wenet_ort_root "${ASR_SDK_WENET_ROOT}/runtime/onnxruntime")
set(_fc_base "${_wenet_ort_root}/fc_base")
set(_gflags_src "${_fc_base}/gflags-src")
set(_glog_src "${_fc_base}/glog-src")
set(_openfst_src "${_fc_base}/openfst-src")

set(GFLAGS_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GFLAGS_BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(GFLAGS_BUILD_gflags_LIB OFF CACHE BOOL "" FORCE)
set(GFLAGS_BUILD_gflags_nothreads_LIB ON CACHE BOOL "" FORCE)
set(GFLAGS_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(GFLAGS_INSTALL_HEADERS OFF CACHE BOOL "" FORCE)
add_subdirectory("${_gflags_src}" "${CMAKE_BINARY_DIR}/_deps/gflags" EXCLUDE_FROM_ALL)

set(WITH_GFLAGS OFF CACHE BOOL "" FORCE)
set(WITH_THREADS ON CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
# Android's bionic headers may expose execinfo.h without providing backtrace().
set(HAVE_EXECINFO_H 0 CACHE INTERNAL "" FORCE)
set(HAVE_STACKTRACE 0 CACHE INTERNAL "" FORCE)
add_subdirectory("${_glog_src}" "${CMAKE_BINARY_DIR}/_deps/glog" EXCLUDE_FROM_ALL)

set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
foreach(_option HAVE_BIN HAVE_SCRIPT HAVE_COMPACT HAVE_COMPRESS HAVE_CONST
                HAVE_FAR HAVE_GRM HAVE_PDT HAVE_MPDT HAVE_LINEAR
                HAVE_LOOKAHEAD HAVE_NGRAM HAVE_PYTHON HAVE_SPECIAL)
  set(${_option} OFF CACHE BOOL "" FORCE)
endforeach()
add_subdirectory("${_openfst_src}" "${CMAKE_BINARY_DIR}/_deps/openfst" EXCLUDE_FROM_ALL)
set_target_properties(fst PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(fst PUBLIC glog gflags_nothreads_static)

add_library(asr_sdk_frontend STATIC
  "${_wenet_ort_root}/frontend/feature_pipeline.cc"
  "${_wenet_ort_root}/frontend/fft.cc")
target_include_directories(asr_sdk_frontend PUBLIC
  "${_wenet_ort_root}"
  "${_openfst_src}/src/include")
target_link_libraries(asr_sdk_frontend PUBLIC fst Threads::Threads)

set(_wetext_root "${CMAKE_CURRENT_SOURCE_DIR}/third_party/WeTextProcessing")
add_library(asr_sdk_wetext_utils STATIC
  "${_wetext_root}/runtime/utils/wetext_string.cc")
target_include_directories(asr_sdk_wetext_utils PUBLIC
  "${_wetext_root}/runtime" "${_openfst_src}/src/include")
target_link_libraries(asr_sdk_wetext_utils PUBLIC fst)

add_library(asr_sdk_wetext_processor STATIC
  "${_wetext_root}/runtime/processor/wetext_processor.cc"
  "${_wetext_root}/runtime/processor/wetext_token_parser.cc")
target_include_directories(asr_sdk_wetext_processor PUBLIC
  "${_wetext_root}/runtime" "${_openfst_src}/src/include")
target_link_libraries(asr_sdk_wetext_processor PUBLIC
  fst asr_sdk_wetext_utils ${CMAKE_DL_LIBS})

# KenLM is built from the source pinned in ASR 0.0.15.
set(_kenlm_root "${CMAKE_CURRENT_SOURCE_DIR}/third_party/kenlm")
set(_kenlm_util_src
  util/bit_packing.cc
  util/double-conversion/bignum-dtoa.cc
  util/double-conversion/bignum.cc
  util/double-conversion/cached-powers.cc
  util/double-conversion/double-to-string.cc
  util/double-conversion/fast-dtoa.cc
  util/double-conversion/fixed-dtoa.cc
  util/double-conversion/string-to-double.cc
  util/double-conversion/strtod.cc
  util/ersatz_progress.cc
  util/exception.cc
  util/file.cc
  util/file_piece.cc
  util/float_to_string.cc
  util/integer_to_string.cc
  util/mmap.cc
  util/murmur_hash.cc
  util/parallel_read.cc
  util/pool.cc
  util/read_compressed.cc
  util/scoped.cc
  util/spaces.cc
  util/string_piece.cc
  util/usage.cc)
list(TRANSFORM _kenlm_util_src PREPEND "${_kenlm_root}/")
add_library(asr_sdk_kenlm_util STATIC ${_kenlm_util_src})
add_library(kenlm::kenlm_util ALIAS asr_sdk_kenlm_util)
target_include_directories(asr_sdk_kenlm_util PUBLIC
  $<BUILD_INTERFACE:${_kenlm_root}>
  $<INSTALL_INTERFACE:include/kenlm>)
target_compile_definitions(asr_sdk_kenlm_util PUBLIC
  _LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
target_link_libraries(asr_sdk_kenlm_util PUBLIC Threads::Threads)

set(_kenlm_lm_src
  lm/bhiksha.cc lm/binary_format.cc lm/config.cc lm/lm_exception.cc
  lm/model.cc lm/quantize.cc lm/read_arpa.cc lm/search_hashed.cc
  lm/search_trie.cc lm/sizes.cc lm/trie.cc lm/trie_sort.cc
  lm/value_build.cc lm/virtual_interface.cc lm/vocab.cc)
list(TRANSFORM _kenlm_lm_src PREPEND "${_kenlm_root}/")
add_library(asr_sdk_kenlm STATIC ${_kenlm_lm_src})
add_library(kenlm::kenlm ALIAS asr_sdk_kenlm)
target_include_directories(asr_sdk_kenlm PUBLIC
  $<BUILD_INTERFACE:${_kenlm_root}>
  $<INSTALL_INTERFACE:include/kenlm>)
target_compile_definitions(asr_sdk_kenlm PUBLIC
  KENLM_MAX_ORDER=6
  _LIBCPP_ENABLE_CXX17_REMOVED_UNARY_BINARY_FUNCTION)
target_link_libraries(asr_sdk_kenlm PUBLIC asr_sdk_kenlm_util Threads::Threads)
install(TARGETS asr_sdk_kenlm_util asr_sdk_kenlm
  EXPORT flashlight-text-targets
  ARCHIVE DESTINATION lib)

set(FL_TEXT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FL_TEXT_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(FL_TEXT_BUILD_PYTHON_PACKAGE OFF CACHE BOOL "" FORCE)
set(FL_TEXT_USE_KENLM ON CACHE BOOL "" FORCE)
set(FL_TEXT_BUILD_STANDALONE ON CACHE BOOL "" FORCE)
set(FL_TEXT_KENLM_MAX_ORDER 6 CACHE STRING "" FORCE)
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/third_party/flashlight-text"
                 "${CMAKE_BINARY_DIR}/_deps/flashlight-text" EXCLUDE_FROM_ALL)

set(_ctc_sources ${ASR_SDK_CTC_SOURCES})
set(_flashlight_sources ${ASR_SDK_FLASHLIGHT_SOURCES})
add_library(asr_sdk_ctc_core STATIC ${_ctc_sources} ${_flashlight_sources})
target_include_directories(asr_sdk_ctc_core PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${_wenet_ort_root}")
target_link_libraries(asr_sdk_ctc_core PUBLIC
  asr_sdk_frontend onnxruntime flashlight-text flashlight-text-kenlm
  PRIVATE ${ASR_SDK_PLATFORM_LIBS})

set(_asr_sources ${ASR_SDK_BASE_SOURCES})
add_library(asr_sdk SHARED ${_asr_sources})
target_include_directories(asr_sdk
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
         $<INSTALL_INTERFACE:include>
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src"
          "${_wenet_ort_root}"
          "${_wetext_root}/runtime")
target_compile_definitions(asr_sdk PRIVATE
  ASR_SDK_BUILDING_LIBRARY=1
  ASR_SDK_PRODUCTION_FLASHLIGHT_ONLY=1
  ASR_SDK_VERSION_STRING="${PROJECT_VERSION}"
  ASR_SDK_ABI_VERSION=5
  ASR_SDK_WENET_COMMIT="production-flashlight-only"
  ASR_SDK_ONNXRUNTIME_VERSION="1.16.3"
  ASR_SDK_FLASHLIGHT_TEXT_COMMIT="49e163ab1e7b8108922512c294ab8513b89f404c"
  ASR_SDK_KENLM_COMMIT="5bf7b46558e1c5595bf3b8c9b0b1f9d8d257040a"
  ASR_SDK_WETEXT_COMMIT="57f85850156c519690f0e60b2096f2a21aa7fa4c")
target_link_libraries(asr_sdk PRIVATE
  asr_sdk_ctc_core asr_sdk_wetext_processor onnxruntime
  ${ASR_SDK_PLATFORM_LIBS})
set_target_properties(asr_sdk PROPERTIES
  OUTPUT_NAME asr_sdk
  CXX_VISIBILITY_PRESET hidden
  C_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON)
target_link_options(asr_sdk PRIVATE "LINKER:--exclude-libs,ALL")

if(ASR_SDK_BUILD_TOOLS)
  add_executable(asr_stream_file
    cli/asr_stream_file.cc src/audio/wav_reader.cc src/utils/timer.cc)
  target_include_directories(asr_stream_file PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(asr_stream_file PRIVATE asr_sdk)

  add_executable(print_build_info cli/print_sdk_info.cc)
  target_link_libraries(print_build_info PRIVATE asr_sdk)
endif()

include(GNUInstallDirs)
install(TARGETS asr_sdk LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/asr_sdk DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
if(ASR_SDK_BUILD_TOOLS)
  install(TARGETS asr_stream_file print_build_info
          RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
