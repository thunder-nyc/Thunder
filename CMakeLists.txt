cmake_minimum_required(VERSION 2.6)
project(Thunder)

# The version number
set(THUNDER_VERSION_MAJOR 0)
set(THUNDER_VERSION_MINOR 0)
set(THUNDER_VERSION_PATCH 0)

# Add -std=c++11 option
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" HAS_CXX11)
if(HAS_CXX11)
  add_compile_options("-std=c++11")
else()
  message(FATAL_ERROR, "The compiler does not support c++ 11")
endif()

# Find boost
find_package(Boost 1.56.0 REQUIRED COMPONENTS serialization)

# Add include directories
include_directories("${CMAKE_SOURCE_DIR}")

# Add all libraries
add_library(thunder_exception thunder/exception/contiguity_error.cpp)
add_library(thunder_storage thunder/storage/storage.cpp)
target_link_libraries(thunder_storage thunder_exception ${Boost_LIBRARIES})
add_library(thunder_tensor thunder/tensor/tensor.cpp)
target_link_libraries(thunder_tensor thunder_exception thunder_storage ${Boost_LIBRARIES})

# Enable testing or not depending on build option
option(BUILD_TESTS "Whether to build tests")
if(BUILD_TESTS)
  enable_testing()
  # Add gtests
  set(GTEST_ROOT "${CMAKE_SOURCE_DIR}/test/gtest-1.7.0")
  add_library(gtest "${GTEST_ROOT}/gtest/gtest-all.cc")
  target_include_directories(gtest PUBLIC "${GTEST_ROOT}")
  add_library(gtest_main "${GTEST_ROOT}/gtest/gtest_main.cc")
  target_include_directories(gtest_main PUBLIC "${GTEST_ROOT}")
  target_link_libraries(gtest_main gtest)
endif()
macro(add_thunder_test name)
  if(BUILD_TESTS)
    add_executable(${name} test/${name}.cpp)
    target_link_libraries(${name} thunder_exception thunder_storage thunder_tensor gtest gtest_main ${Boost_LIBRARIES})
    add_test(${name} ${name})
  endif()
endmacro()
add_thunder_test(storage)
add_thunder_test(index_iterator)
add_thunder_test(tensor_access)
add_thunder_test(tensor_modify)
add_thunder_test(tensor_type)
add_thunder_test(tensor_unary)
add_thunder_test(tensor_binary)
add_thunder_test(tensor_ternary)
add_thunder_test(tensor_reduction)
add_thunder_test(tensor_static)
add_thunder_test(tensor_operator)
add_thunder_test(tensor_serialize)
add_thunder_test(complex_unary)
add_thunder_test(complex_binary)
add_thunder_test(complex_ternary)
add_thunder_test(complex_reduction)
add_thunder_test(complex_operator)
