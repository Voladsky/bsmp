file(GLOB_RECURSE bsmp_SRC "*.cpp" "*.hpp" "*.cu" "*.cuh")

add_library(bsmp STATIC
    ${bsmp_SRC}
)

target_include_directories(bsmp PUBLIC
    ${CMAKE_SOURCE_DIR}/include # for public headers
    ${CMAKE_CURRENT_SOURCE_DIR} # if private headers will be needed
)

target_compile_definitions(bsmp PUBLIC BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

if(CUDAToolkit_FOUND)
  target_link_libraries(bsmp PUBLIC CUDA::cudart CUDA::cublas)
else()
  message(STATUS "CUDAToolkit not found via package, relying on CUDA language cudart")
  # When CUDA is enabled as a language, the cudart library is available via cuda runtime library
  # On most systems linking is automatic, but we keep explicit to be safe when available
endif()

target_link_libraries(bsmp PRIVATE nlohmann_json)

# Now all targets should be setup with function from Setup.cmake
bsmp_setup_target(bsmp)
