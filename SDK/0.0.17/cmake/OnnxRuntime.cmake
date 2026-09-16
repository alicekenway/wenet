if(NOT ASR_SDK_DYNAMIC_ONNXRUNTIME)
  message(FATAL_ERROR "ASR_SDK_DYNAMIC_ONNXRUNTIME=OFF is not supported")
endif()

if(NOT ASR_SDK_REQUIRED_ORT_VERSION MATCHES "^1\\.([0-9]+)\\.[0-9]+$")
  message(FATAL_ERROR "ASR requires an explicit ONNX Runtime 1.x.y version")
endif()
set(_selected_ort_api "${CMAKE_MATCH_1}")

if(NOT EXISTS "${ASR_SDK_ONNXRUNTIME_ROOT}")
  message(FATAL_ERROR
    "ONNX Runtime root not found: ${ASR_SDK_ONNXRUNTIME_ROOT}\n"
    "Expected third_party/onnxruntime with include/ and lib/.")
endif()

set(ASR_SDK_ONNXRUNTIME_INCLUDE_DIR
    "${ASR_SDK_ONNXRUNTIME_ROOT}/include")
if(NOT EXISTS "${ASR_SDK_ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
  message(FATAL_ERROR
    "ONNX Runtime C++ header not found: "
    "${ASR_SDK_ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_cxx_api.h")
endif()
if(NOT EXISTS "${ASR_SDK_ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")
  message(FATAL_ERROR
    "ONNX Runtime C header not found: "
    "${ASR_SDK_ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")
endif()

file(STRINGS
  "${ASR_SDK_ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h"
  _ort_api_line REGEX "^#define ORT_API_VERSION [0-9]+$")
set(_expected_ort_api "${_selected_ort_api}")
if(_expected_ort_api AND
   NOT _ort_api_line STREQUAL "#define ORT_API_VERSION ${_expected_ort_api}")
  message(FATAL_ERROR
    "ONNX Runtime header mismatch: ${ASR_SDK_REQUIRED_ORT_VERSION} requires "
    "API ${_expected_ort_api}, found '${_ort_api_line}'")
endif()

set(_ort_versioned_lib
    "${ASR_SDK_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.${ASR_SDK_REQUIRED_ORT_VERSION}")
if(EXISTS "${_ort_versioned_lib}")
  set(ASR_SDK_ONNXRUNTIME_LIB "${_ort_versioned_lib}")
  set(ASR_SDK_ONNXRUNTIME_VERSION "${ASR_SDK_REQUIRED_ORT_VERSION}")
else()
  message(FATAL_ERROR
    "Exact ONNX Runtime library not found: ${_ort_versioned_lib}")
endif()

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ASR_SDK_ONNXRUNTIME_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${ASR_SDK_ONNXRUNTIME_INCLUDE_DIR}"
)

if(UNIX AND NOT APPLE)
  file(CREATE_LINK "${ASR_SDK_ONNXRUNTIME_LIB}"
       "${CMAKE_BINARY_DIR}/libonnxruntime.so.1"
       SYMBOLIC RESULT _ort_link_soname)
  file(CREATE_LINK "${ASR_SDK_ONNXRUNTIME_LIB}"
       "${CMAKE_BINARY_DIR}/libonnxruntime.so"
       SYMBOLIC RESULT _ort_link_plain)
endif()

message(STATUS "ASR SDK ONNX Runtime root: ${ASR_SDK_ONNXRUNTIME_ROOT}")
message(STATUS "ASR SDK ONNX Runtime: ${ASR_SDK_ONNXRUNTIME_LIB}")
message(STATUS "ASR SDK ONNX Runtime version: ${ASR_SDK_ONNXRUNTIME_VERSION}")
message(STATUS "ASR SDK ONNX Runtime linkage: dynamic")
