set(GTEST_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/fused-src)

add_library(gtest STATIC "${GTEST_ROOT}/gtest/gtest-all.cc" "${GTEST_ROOT}/gtest/gtest.h")
target_include_directories(gtest PUBLIC "${GTEST_ROOT}")
target_link_libraries(gtest ${CMAKE_THREAD_LIBS_INIT})

add_library(gtest_main STATIC "${GTEST_ROOT}/gtest/gtest_main.cc" "${GTEST_ROOT}/gtest/gtest.h")
target_include_directories(gtest_main PUBLIC "${GTEST_ROOT}")
target_link_libraries(gtest_main gtest)
