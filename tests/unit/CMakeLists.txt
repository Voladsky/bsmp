set(CMAKE_FOLDER "tests/unit")

file(COPY ${CMAKE_SOURCE_DIR}/data/schemas/v1 DESTINATION
${CMAKE_CURRENT_BINARY_DIR}/data/schemas)

add_custom_target(unit_tests)

bsmp_add_unit_test(NAME test_json_recorder SOURCES test_json_recorder.cpp LIBS bsmp nlohmann_json::nlohmann_json
        nlohmann_json_schema_validator)

add_dependencies(unit_tests test_json_recorder)

# Framework unit tests. All live in the common `spmv` namespace (see
# tests/support/spmv_test_utils.hpp) so future framework functionality can
# reuse the same helpers.
set(SPMV_UNIT_TESTS
    test_static_asserts
    test_parsers
    test_allocators
    test_device_buffer
    test_dense_vector
    test_workspace
    test_profiling
    test_spmv_matrix
)

foreach(test_name IN LISTS SPMV_UNIT_TESTS)
    bsmp_add_unit_test(NAME ${test_name} SOURCES ${test_name}.cpp)
    target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/tests/support)
    add_dependencies(unit_tests ${test_name})
endforeach()

# test_spmv_matrix launches real kernels (SparseMatrix<COOFormat>::multiply),
# so it needs the device-link anchor: bsmp is built with separable compilation
# and CMake only device-links executables that contain a .cu translation unit.
target_sources(test_spmv_matrix PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/support/device_link_stub.cu)
