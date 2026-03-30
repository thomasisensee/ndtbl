cmake_minimum_required(VERSION 3.23)

# Set a name and a version number for your project:
project(ndtbl
  VERSION 0.0.1
  LANGUAGES CXX
)

# Initialize some default paths
include(GNUInstallDirs)

# Define build options
option(ndtbl_BUILD_TESTING "Enable building of tests" OFF)

if(PROJECT_IS_TOP_LEVEL)
  option(ndtbl_BUILD_DOCS "Enable building of documentation" ON)
else()
  option(ndtbl_BUILD_DOCS "Enable building of documentation" OFF)
endif()

# Add an interface target for the header-only library
add_library(ndtbl INTERFACE)

# Add the public headers that belong to this library
target_sources(ndtbl
  INTERFACE
    FILE_SET HEADERS
    BASE_DIRS include
    FILES
      include/ndtbl/types.hpp
      include/ndtbl/axis.hpp
      include/ndtbl/grid.hpp
      include/ndtbl/field_group.hpp
      include/ndtbl/any_field_group.hpp
      include/ndtbl/metadata.hpp
      include/ndtbl/io.hpp
      include/ndtbl/ndtbl.hpp
      include/ndtbl/detail/binary_io.hpp
)

# Request a minimum C++ standard for this target
target_compile_features(ndtbl
  INTERFACE
    cxx_std_14
)

if(PROJECT_IS_TOP_LEVEL)
  # Compile the application executable
  add_subdirectory(app)
endif()

if(PROJECT_IS_TOP_LEVEL)
  # Compile the tests
  include(CTest)
  if(ndtbl_BUILD_TESTING)
    add_subdirectory(tests)
  endif()
endif()

# Add the documentation
if(PROJECT_IS_TOP_LEVEL AND ndtbl_BUILD_DOCS)
  add_subdirectory(doc)
endif()

# Add an alias target for use if this project is included as a subproject in another project
add_library(ndtbl::ndtbl ALIAS ndtbl)

# Install the target and headers
install(
  TARGETS ndtbl
  EXPORT ndtblTargets
  FILE_SET HEADERS
)

# Install the exported targets (creates ndtblTargets.cmake)
install(
  EXPORT ndtblTargets
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ndtbl
  NAMESPACE ndtbl::
)

# Include helpers for calling configure_package_config_file
include(CMakePackageConfigHelpers)

# Generate the ndtblConfig.cmake file
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/ndtblConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/ndtblConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ndtbl
)

# Install the generated Config file
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ndtblConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ndtbl
)

# Generate a version file
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/ndtblConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)

# Install the version file
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/ndtblConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ndtbl
)

# Print a summary of found dependencies
include(FeatureSummary)
feature_summary(WHAT ALL)
