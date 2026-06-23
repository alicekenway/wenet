set(_ONNXRUNTIME_HINTS
  ${ONNXRuntime_ROOT}
  $ENV{ONNXRuntime_ROOT}
  $ENV{CONDA_PREFIX})

find_path(ONNXRuntime_INCLUDE_DIR
  NAMES onnxruntime_cxx_api.h
  HINTS ${_ONNXRUNTIME_HINTS}
  PATH_SUFFIXES include include/onnxruntime)

find_library(ONNXRuntime_LIBRARY
  NAMES onnxruntime
  HINTS ${_ONNXRUNTIME_HINTS}
  PATH_SUFFIXES lib lib64)

if(NOT ONNXRuntime_LIBRARY)
  foreach(_root IN LISTS _ONNXRUNTIME_HINTS)
    if(_root)
      file(GLOB _onnxruntime_python_libs
        "${_root}/lib/python*/site-packages/onnxruntime/capi/libonnxruntime.so*")
      list(SORT _onnxruntime_python_libs)
      list(REVERSE _onnxruntime_python_libs)
      list(LENGTH _onnxruntime_python_libs _onnxruntime_python_libs_len)
      if(_onnxruntime_python_libs_len GREATER 0)
        list(GET _onnxruntime_python_libs 0 ONNXRuntime_LIBRARY)
        if(ONNXRuntime_LIBRARY)
          break()
        endif()
      endif()
    endif()
  endforeach()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
  REQUIRED_VARS ONNXRuntime_INCLUDE_DIR ONNXRuntime_LIBRARY)

if(ONNXRuntime_FOUND AND NOT TARGET ONNXRuntime::ONNXRuntime)
  add_library(ONNXRuntime::ONNXRuntime UNKNOWN IMPORTED)
  set_target_properties(ONNXRuntime::ONNXRuntime PROPERTIES
    IMPORTED_LOCATION "${ONNXRuntime_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${ONNXRuntime_INCLUDE_DIR}")
endif()
