# Declare a new library target
add_library(ndtbl)

# Add the implementation file and mark the public headers that belong to this library
target_sources(ndtbl
  PRIVATE
    ndtbl.cpp

  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ../include
    FILES
      ../include/ndtbl/ndtbl.hpp
)

# Request a minimum C++ standard for this target
target_compile_features(ndtbl
  PUBLIC
    cxx_std_14
)
