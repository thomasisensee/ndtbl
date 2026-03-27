# Add convertet tool executable
add_executable(ndtbl_convert
  ndtbl_convert.cpp
)

target_link_libraries(ndtbl_convert
  PRIVATE
    ndtbl
)

target_compile_features(ndtbl_convert
  PRIVATE
    cxx_std_20
)

set_target_properties(ndtbl_convert
  PROPERTIES
    OUTPUT_NAME "ndtbl-convert"
)

# Add inspect tool executable
add_executable(ndtbl_inspect
  ndtbl_inspect.cpp
)

target_link_libraries(ndtbl_inspect
  PRIVATE
    ndtbl
)

target_compile_features(ndtbl_inspect
  PRIVATE
    cxx_std_20
)

set_target_properties(ndtbl_inspect
  PROPERTIES
    OUTPUT_NAME "ndtbl-inspect"
)
