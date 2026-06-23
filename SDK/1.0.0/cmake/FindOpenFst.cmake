find_path(OpenFst_INCLUDE_DIR
  NAMES fst/fstlib.h)

find_library(OpenFst_LIBRARY
  NAMES fst openfst)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenFst
  REQUIRED_VARS OpenFst_INCLUDE_DIR OpenFst_LIBRARY)

if(OpenFst_FOUND AND NOT TARGET OpenFst::fst)
  add_library(OpenFst::fst UNKNOWN IMPORTED)
  set_target_properties(OpenFst::fst PROPERTIES
    IMPORTED_LOCATION "${OpenFst_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${OpenFst_INCLUDE_DIR}")
endif()
