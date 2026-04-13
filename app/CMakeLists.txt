
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
