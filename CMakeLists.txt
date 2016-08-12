cmake_minimum_required(VERSION 2.6)
project(Thunder)

# Enable Fortran for blas/lapack interfaces
enable_language(Fortran)

# The version number
set(THUNDER_VERSION_MAJOR 0)
set(THUNDER_VERSION_MINOR 3)
set(THUNDER_VERSION_PATCH 0)

# By default, build shared library
option(BUILD_SHARED_LIBS "Whether to build shared libraries" ON)

# By default, build in release mode
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Add -std=c++11 option
include(CheckCXXCompilerFlag)
if (UNIX)
  check_cxx_compiler_flag("-std=c++11" HAS_CXX11)
  if(HAS_CXX11)
    # For compatibility. Only CMake 2.8.12+ provides add_compile_options("-std=c++11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  else()
    message(FATAL_ERROR "The compiler does not support C++ 11")
  endif()
endif()

# Find BLAS
find_package(BLAS REQUIRED)

# Find pthread
set(CMAKE_THREAD_PREFER_PTHREAD true)
find_package(Threads)

# Enable testing or not depending on build option
option(BUILD_THUNDER_TESTS "Whether to build tests for thunder")
if(BUILD_THUNDER_TESTS)
  if(CMAKE_USE_PTHREADS_INIT)
    enable_testing()
  else()
    message(WARNING "Could not find pthread. Tests disabled.")
    set(BUILD_THUNDER_TESTS OFF)
  endif()
endif()

# Add packages
add_subdirectory(packages)

# Add tools
add_subdirectory(tools)
