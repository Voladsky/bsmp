set(CMAKE_FOLDER "tests/smoke")

add_custom_target(smoke_tests)

# gtest-based integration smoke tests. Mirrors bsmp_add_unit_test but
# registers discovered tests under the "smoke." prefix.
function(bsmp_add_framework_smoke_test test_name source_file)
    add_executable(${test_name} ${source_file})

    target_link_libraries(${test_name} PRIVATE bsmp)
    target_compile_definitions(${test_name} PRIVATE BSMP_BLOCK_SIZE=${BSMP_BLOCK_SIZE})

    bsmp_setup_target(${test_name} DISABLE_MEMCHECK)

    target_include_directories(${test_name} PRIVATE
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/tests/support
    )

    target_link_libraries(${test_name} PRIVATE GTest::gtest_main GTest::gmock)

    gtest_discover_tests(${test_name}
        PREFIX "smoke."
    )

    add_dependencies(smoke_tests ${test_name})
endfunction()

bsmp_add_framework_smoke_test(spmv_smoke test_spmv_smoke.cpp)

# The smoke test launches real kernels via SparseMatrix<COOFormat>::multiply;
# with separable compilation the executable needs a .cu translation unit to
# trigger CMake's device-link step (see device_link_stub.cu).
target_sources(spmv_smoke PRIVATE
    ${CMAKE_SOURCE_DIR}/tests/support/device_link_stub.cu)