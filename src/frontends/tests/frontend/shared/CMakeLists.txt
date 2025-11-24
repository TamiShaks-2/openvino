# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

set(TARGET_NAME "frontend_shared_test_classes")

file(GLOB_RECURSE LIBRARY_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE LIBRARY_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp)

add_library(${TARGET_NAME} STATIC EXCLUDE_FROM_ALL ${LIBRARY_SRC} ${LIBRARY_HEADERS})

add_subdirectory(gtest_main_manifest)
add_subdirectory(test_builtin_extensions)

target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${TARGET_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)

target_link_libraries(${TARGET_NAME}
        PUBLIC
                openvino::offline_transformations
                common_test_utils
                func_test_utils
                openvino::util
                openvino::runtime
        PRIVATE
                openvino::cnpy)

add_dependencies(${TARGET_NAME} test_builtin_extensions)

# build a test binary for this source file
add_executable(multi_dot_opt_test src/multi_dot_opt_test.cpp)

# link the test binary to the static library and the gtest main target
target_link_libraries(multi_dot_opt_test
        PRIVATE
                frontend_shared_test_classes
                gtest_main
)

# register the test so ctest can find it
add_test(NAME multi_dot_opt_test COMMAND multi_dot_opt_test)

target_compile_definitions(${TARGET_NAME}
        PRIVATE
                SHARED_LIB_PREFIX="${CMAKE_SHARED_LIBRARY_PREFIX}"
                SHARED_LIB_SUFFIX="${OV_BUILD_POSTFIX}${CMAKE_SHARED_LIBRARY_SUFFIX}"
)

ov_add_clang_format_target(${TARGET_NAME}_clang FOR_TARGETS ${TARGET_NAME})

ov_build_target_faster(${TARGET_NAME} PCH)
message("🐎🐆this is cmake of the culc tests")