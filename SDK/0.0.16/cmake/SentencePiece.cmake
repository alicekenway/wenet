# Pinned to google/sentencepiece v0.2.1, commit
# 31646a467d2051eb904e0b45de3a73e91fe1c1e3 (Apache-2.0).
# EXCLUDE_FROM_ALL keeps trainers and command-line tools out of SDK builds.
set(_spm_root "${CMAKE_CURRENT_SOURCE_DIR}/third_party/sentencepiece")
if(NOT EXISTS "${_spm_root}/src/sentencepiece_processor.h")
  message(FATAL_ERROR "Missing pinned SentencePiece v0.2.1 sources: ${_spm_root}")
endif()
set(SPM_ENABLE_SHARED OFF CACHE BOOL "" FORCE)
set(SPM_ENABLE_TCMALLOC OFF CACHE BOOL "" FORCE)
set(SPM_BUILD_TEST OFF CACHE BOOL "" FORCE)
add_subdirectory("${_spm_root}" "${CMAKE_BINARY_DIR}/_deps/sentencepiece" EXCLUDE_FROM_ALL)
set_target_properties(sentencepiece-static PROPERTIES
  POSITION_INDEPENDENT_CODE ON CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN YES)
target_include_directories(sentencepiece-static INTERFACE "${_spm_root}/src")
add_library(asr_sdk::sentencepiece ALIAS sentencepiece-static)
