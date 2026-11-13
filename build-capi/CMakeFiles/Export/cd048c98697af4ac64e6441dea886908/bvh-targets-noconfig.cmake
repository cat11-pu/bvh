#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "bvh::v2::bvh_c" for configuration ""
set_property(TARGET bvh::v2::bvh_c APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(bvh::v2::bvh_c PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libbvh_c.so"
  IMPORTED_SONAME_NOCONFIG "libbvh_c.so"
  )

list(APPEND _cmake_import_check_targets bvh::v2::bvh_c )
list(APPEND _cmake_import_check_files_for_bvh::v2::bvh_c "${_IMPORT_PREFIX}/lib/libbvh_c.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
