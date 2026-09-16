if(NOT ASR_SDK_ENABLE_FLASHLIGHT_DECODER)
  return()
endif()

if(NOT EXISTS "${ASR_SDK_FLASHLIGHT_TEXT_ROOT}/CMakeLists.txt")
  message(FATAL_ERROR
    "Pinned Flashlight-Text source not found: "
    "${ASR_SDK_FLASHLIGHT_TEXT_ROOT}")
endif()

set(ASR_SDK_FLASHLIGHT_TEXT_COMMIT
    "49e163ab1e7b8108922512c294ab8513b89f404c")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(FL_TEXT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FL_TEXT_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
set(FL_TEXT_BUILD_PYTHON_PACKAGE OFF CACHE BOOL "" FORCE)
set(FL_TEXT_USE_KENLM ON CACHE BOOL "" FORCE)
set(FL_TEXT_BUILD_STANDALONE ON CACHE BOOL "" FORCE)
set(FL_TEXT_KENLM_MAX_ORDER 6 CACHE STRING "" FORCE)

add_subdirectory("${ASR_SDK_FLASHLIGHT_TEXT_ROOT}"
                 "${CMAKE_BINARY_DIR}/_deps/flashlight-text-build"
                 EXCLUDE_FROM_ALL)

add_library(asr_sdk_flashlight_text INTERFACE)
target_link_libraries(asr_sdk_flashlight_text INTERFACE
  flashlight::flashlight-text
  flashlight::flashlight-text-kenlm
)
add_library(asr_sdk::flashlight_text ALIAS asr_sdk_flashlight_text)

message(STATUS
  "ASR SDK Flashlight-Text source: ${ASR_SDK_FLASHLIGHT_TEXT_ROOT}")
message(STATUS
  "ASR SDK Flashlight-Text commit: ${ASR_SDK_FLASHLIGHT_TEXT_COMMIT}")
