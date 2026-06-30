if(NOT ASR_SDK_ENABLE_FLASHLIGHT_DECODER)
  return()
endif()

if(NOT EXISTS "${ASR_SDK_KENLM_ROOT}/CMakeLists.txt")
  message(FATAL_ERROR
    "Pinned KenLM source not found: ${ASR_SDK_KENLM_ROOT}")
endif()

set(ASR_SDK_KENLM_COMMIT
    "5bf7b46558e1c5595bf3b8c9b0b1f9d8d257040a")

find_package(Threads REQUIRED)
find_package(ZLIB QUIET)
find_package(BZip2 QUIET)
find_package(LibLZMA QUIET)

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
  util/usage.cc
)
list(TRANSFORM _kenlm_util_src PREPEND "${ASR_SDK_KENLM_ROOT}/")

add_library(asr_sdk_kenlm_util STATIC ${_kenlm_util_src})
add_library(kenlm::kenlm_util ALIAS asr_sdk_kenlm_util)
target_include_directories(asr_sdk_kenlm_util PUBLIC
  $<BUILD_INTERFACE:${ASR_SDK_KENLM_ROOT}>
  $<INSTALL_INTERFACE:include/kenlm>
)
target_link_libraries(asr_sdk_kenlm_util PUBLIC Threads::Threads)
if(UNIX AND NOT APPLE)
  target_link_libraries(asr_sdk_kenlm_util PUBLIC rt)
endif()
if(TARGET ZLIB::ZLIB)
  target_compile_definitions(asr_sdk_kenlm_util PRIVATE HAVE_ZLIB)
  target_link_libraries(asr_sdk_kenlm_util PUBLIC ZLIB::ZLIB)
endif()
if(TARGET BZip2::BZip2)
  target_compile_definitions(asr_sdk_kenlm_util PRIVATE HAVE_BZLIB)
  target_link_libraries(asr_sdk_kenlm_util PUBLIC BZip2::BZip2)
endif()
if(TARGET LibLZMA::LibLZMA)
  target_compile_definitions(asr_sdk_kenlm_util PRIVATE HAVE_XZLIB)
  target_link_libraries(asr_sdk_kenlm_util PUBLIC LibLZMA::LibLZMA)
endif()
set_target_properties(asr_sdk_kenlm_util PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

set(_kenlm_lm_src
  lm/bhiksha.cc
  lm/binary_format.cc
  lm/config.cc
  lm/lm_exception.cc
  lm/model.cc
  lm/quantize.cc
  lm/read_arpa.cc
  lm/search_hashed.cc
  lm/search_trie.cc
  lm/sizes.cc
  lm/trie.cc
  lm/trie_sort.cc
  lm/value_build.cc
  lm/virtual_interface.cc
  lm/vocab.cc
)
list(TRANSFORM _kenlm_lm_src PREPEND "${ASR_SDK_KENLM_ROOT}/")

add_library(asr_sdk_kenlm STATIC ${_kenlm_lm_src})
add_library(kenlm::kenlm ALIAS asr_sdk_kenlm)
target_include_directories(asr_sdk_kenlm PUBLIC
  $<BUILD_INTERFACE:${ASR_SDK_KENLM_ROOT}>
  $<INSTALL_INTERFACE:include/kenlm>
)
target_compile_definitions(asr_sdk_kenlm PUBLIC KENLM_MAX_ORDER=6)
target_link_libraries(asr_sdk_kenlm PUBLIC
  kenlm::kenlm_util
  Threads::Threads
)
set_target_properties(asr_sdk_kenlm PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS asr_sdk_kenlm_util asr_sdk_kenlm
  EXPORT flashlight-text-targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

message(STATUS "ASR SDK KenLM source: ${ASR_SDK_KENLM_ROOT}")
message(STATUS "ASR SDK KenLM tools: ${ASR_SDK_KENLM_INSTALL_ROOT}/bin")
message(STATUS "ASR SDK KenLM commit: ${ASR_SDK_KENLM_COMMIT}")
