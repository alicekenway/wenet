if(NOT ASR_SDK_DYNAMIC_ONNXRUNTIME)
  message(FATAL_ERROR "ASR_SDK_DYNAMIC_ONNXRUNTIME=OFF is not supported")
endif()

if(NOT EXISTS "${ASR_SDK_ONNXRUNTIME_ROOT}")
  message(FATAL_ERROR
    "ONNX Runtime root not found: ${ASR_SDK_ONNXRUNTIME_ROOT}\n"
    "Expected third_party/onnxruntime with include/ and lib/, or an "
    "extracted Android AAR with headers/ and jni/<abi>/.")
endif()

# Headers.
if(ANDROID AND EXISTS
   "${ASR_SDK_ONNXRUNTIME_ROOT}/headers/onnxruntime_cxx_api.h")
  set(ASR_SDK_ONNXRUNTIME_INCLUDE_DIR
      "${ASR_SDK_ONNXRUNTIME_ROOT}/headers")
elseif(EXISTS "${ASR_SDK_ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
  set(ASR_SDK_ONNXRUNTIME_INCLUDE_DIR
      "${ASR_SDK_ONNXRUNTIME_ROOT}/include")
else()
  message(FATAL_ERROR
    "ONNX Runtime headers not found. Expected either:\n"
    "  ${ASR_SDK_ONNXRUNTIME_ROOT}/headers/onnxruntime_cxx_api.h\n"
    "  ${ASR_SDK_ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
endif()
if(NOT EXISTS "${ASR_SDK_ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")
  message(FATAL_ERROR
    "ONNX Runtime C header not found: "
    "${ASR_SDK_ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")
endif()

# Library.
if(ANDROID)
  set(_ort_android_aar_lib
      "${ASR_SDK_ONNXRUNTIME_ROOT}/jni/${ANDROID_ABI}/libonnxruntime.so")
  set(_ort_android_normalized_lib
      "${ASR_SDK_ONNXRUNTIME_ROOT}/lib/${ANDROID_ABI}/libonnxruntime.so")

  if(EXISTS "${_ort_android_aar_lib}")
    set(ASR_SDK_ONNXRUNTIME_LIB "${_ort_android_aar_lib}")
  elseif(EXISTS "${_ort_android_normalized_lib}")
    set(ASR_SDK_ONNXRUNTIME_LIB "${_ort_android_normalized_lib}")
  else()
    message(FATAL_ERROR
      "Android ONNX Runtime library not found for ABI ${ANDROID_ABI}. "
      "Expected either:\n"
      "  ${_ort_android_aar_lib}\n"
      "  ${_ort_android_normalized_lib}")
  endif()
  set(ASR_SDK_ONNXRUNTIME_VERSION "android-aar")
else()
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
endif()

if(NOT ANDROID
   AND NOT ASR_SDK_ONNXRUNTIME_VERSION STREQUAL ASR_SDK_REQUIRED_ORT_VERSION
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

if(NOT ANDROID AND UNIX AND NOT APPLE)
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
