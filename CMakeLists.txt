cmake_minimum_required (VERSION 2.6)
project (Thunder)

# The version number
set (THUNDER_VERSION_MAJOR 0)
set (THUNDER_VERSION_MINOR 0)
set (THUNDER_VERSION_PATCH 0)

# Add -std=c++11 option
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" HAS_CXX11)
if (HAS_CXX11)
  add_compile_options("-std=c++11")
else ()
  message(FATAL_ERROR, "The compiler does not support c++ 11")
endif ()

# Find boost
find_package(Boost 1.56.0 REQUIRED COMPONENTS serialization)

# Add include directories
include_directories("${CMAKE_SOURCE_DIR}")

# Add 2 libraries
add_library(thunder_exception thunder/exception/contiguity_error.cpp)
add_library(thunder_storage thunder/storage/storage.cpp)
target_link_libraries(thunder_storage thunder_exception ${Boost_LIBRARIES})

# Add gtests
set(GTEST_ROOT "${CMAKE_SOURCE_DIR}/test/gtest-1.7.0")
add_library(gtest "${GTEST_ROOT}/gtest/gtest-all.cc")
target_include_directories(gtest PUBLIC "${GTEST_ROOT}")
add_library(gtest_main "${GTEST_ROOT}/gtest/gtest_main.cc")
target_include_directories(gtest_main PUBLIC "${GTEST_ROOT}")
target_link_libraries(gtest_main gtest)

# Add 1 test binary
add_executable(storage test/storage.cpp)
target_link_libraries(storage thunder_exception thunder_storage gtest gtest_main ${Boost_LIBRARIES})
