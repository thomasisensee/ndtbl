# Declare an executable target
add_executable(ndtbl_app)

# Add source files to the executable target
target_sources(ndtbl_app
  PRIVATE
    ndtbl_app.cpp
)

# Link the executable against the library
target_link_libraries(ndtbl_app
  PRIVATE
    ndtbl
)
