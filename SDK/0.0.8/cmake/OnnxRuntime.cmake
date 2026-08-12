if(NOT ASR_SDK_DYNAMIC_ONNXRUNTIME)
  message(FATAL_ERROR "ASR_SDK_DYNAMIC_ONNXRUNTIME=OFF is not supported")
endif()

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

set(_ort_versioned_lib
    "${ASR_SDK_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.${ASR_SDK_REQUIRED_ORT_VERSION}")
set(_ort_soname_lib "${ASR_SDK_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1")
set(_ort_plain_lib "${ASR_SDK_ONNXRUNTIME_ROOT}/lib/libonnxruntime.so")
if(EXISTS "${_ort_versioned_lib}")
  set(ASR_SDK_ONNXRUNTIME_LIB "${_ort_versioned_lib}")
  set(ASR_SDK_ONNXRUNTIME_VERSION "${ASR_SDK_REQUIRED_ORT_VERSION}")
elseif(EXISTS "${_ort_soname_lib}")
  set(ASR_SDK_ONNXRUNTIME_LIB "${_ort_soname_lib}")
  set(ASR_SDK_ONNXRUNTIME_VERSION "unknown")
elseif(EXISTS "${_ort_plain_lib}")
  set(ASR_SDK_ONNXRUNTIME_LIB "${_ort_plain_lib}")
  set(ASR_SDK_ONNXRUNTIME_VERSION "unknown")
else()
  message(FATAL_ERROR
    "ONNX Runtime library not found under "
    "${ASR_SDK_ONNXRUNTIME_ROOT}/lib. Expected libonnxruntime.so.")
endif()

if(NOT ASR_SDK_ONNXRUNTIME_VERSION STREQUAL ASR_SDK_REQUIRED_ORT_VERSION
   AND NOT ASR_SDK_ALLOW_ORT_VERSION_MISMATCH)
  message(FATAL_ERROR
    "ONNX Runtime version mismatch. Required "
    "${ASR_SDK_REQUIRED_ORT_VERSION}, detected "
    "${ASR_SDK_ONNXRUNTIME_VERSION}. Set "
    "ASR_SDK_ALLOW_ORT_VERSION_MISMATCH=ON only for local experiments.")
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
