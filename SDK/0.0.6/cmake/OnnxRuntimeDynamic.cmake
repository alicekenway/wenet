if(NOT ASR_SDK_DYNAMIC_ONNXRUNTIME)
  message(FATAL_ERROR "ASR_SDK_DYNAMIC_ONNXRUNTIME=OFF is not supported by this SDK plan")
endif()

if(NOT EXISTS "${ASR_SDK_ONNXRUNTIME_LIB}")
  message(FATAL_ERROR "ONNX Runtime shared library not found: ${ASR_SDK_ONNXRUNTIME_LIB}")
endif()

if(NOT EXISTS "${ASR_SDK_ONNXRUNTIME_INCLUDE_DIR}/onnxruntime_c_api.h")
  message(FATAL_ERROR "ONNX Runtime headers not found: ${ASR_SDK_ONNXRUNTIME_INCLUDE_DIR}")
endif()

get_filename_component(ASR_SDK_ONNXRUNTIME_LIB_DIR "${ASR_SDK_ONNXRUNTIME_LIB}" DIRECTORY)
get_filename_component(ASR_SDK_ONNXRUNTIME_LIB_NAME "${ASR_SDK_ONNXRUNTIME_LIB}" NAME)

add_library(onnxruntime SHARED IMPORTED GLOBAL)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION "${ASR_SDK_ONNXRUNTIME_LIB}"
  INTERFACE_INCLUDE_DIRECTORIES "${ASR_SDK_ONNXRUNTIME_INCLUDE_DIR}"
)

set(ASR_SDK_ONNXRUNTIME_VERSION "1.23.2")

# The Linux ONNX Runtime library has SONAME libonnxruntime.so.1. The conda
# package here only exposes the fully versioned file, so create build-tree
# symlinks for RPATH=$ORIGIN testing without mutating the conda environment.
if(UNIX AND NOT APPLE)
  file(CREATE_LINK "${ASR_SDK_ONNXRUNTIME_LIB}"
       "${CMAKE_BINARY_DIR}/libonnxruntime.so.1"
       SYMBOLIC)
  file(CREATE_LINK "${ASR_SDK_ONNXRUNTIME_LIB}"
       "${CMAKE_BINARY_DIR}/libonnxruntime.so"
       SYMBOLIC)
endif()

message(STATUS "ASR SDK ONNX Runtime: ${ASR_SDK_ONNXRUNTIME_LIB}")
message(STATUS "ASR SDK ONNX Runtime linkage: dynamic")
