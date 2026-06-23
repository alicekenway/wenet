find_path(PortAudio_INCLUDE_DIR
  NAMES portaudio.h
  HINTS ${PortAudio_ROOT} $ENV{PortAudio_ROOT} $ENV{CONDA_PREFIX}
  PATH_SUFFIXES include)

find_library(PortAudio_LIBRARY
  NAMES portaudio
  HINTS ${PortAudio_ROOT} $ENV{PortAudio_ROOT} $ENV{CONDA_PREFIX}
  PATH_SUFFIXES lib lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PortAudio
  REQUIRED_VARS PortAudio_INCLUDE_DIR PortAudio_LIBRARY)

if(PortAudio_FOUND AND NOT TARGET PortAudio::PortAudio)
  add_library(PortAudio::PortAudio UNKNOWN IMPORTED)
  set_target_properties(PortAudio::PortAudio PROPERTIES
    IMPORTED_LOCATION "${PortAudio_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${PortAudio_INCLUDE_DIR}")
endif()
