cmake_minimum_required(VERSION 2.6)
project(Thunder)

# The version number
set(THUNDER_VERSION_MAJOR 0)
set(THUNDER_VERSION_MINOR 1)
set(THUNDER_VERSION_PATCH 0)

# By default, build shared library
option(BUILD_SHARED_LIBS "Whether to build shared libraries" ON)

# By default, build in release mode
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

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

# Enable testing or not depending on build option
option(BUILD_THUNDER_TESTS "Whether to build tests for thunder")
if(BUILD_THUNDER_TESTS)
  enable_testing()
endif()

# Add packages
add_subdirectory(packages)

# Add tools
add_subdirectory(tools)
