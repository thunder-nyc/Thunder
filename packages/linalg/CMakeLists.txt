# Get all the include and source files
file(GLOB_RECURSE HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "include/thunder/*.hpp")
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "src/*.cpp")
file(GLOB_RECURSE FORTRAN_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "src/*.f")
file(GLOB_RECURSE TESTS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "test/*.cpp")

# Create the library
add_library(thunder_linalg ${HEADERS} ${SOURCES} ${FORTRAN_SOURCES})
target_include_directories(thunder_linalg PUBLIC "include")
target_link_libraries(thunder_linalg thunder_exception thunder_storage thunder_tensor ${BLAS_LIBRARIES})

# Create installation
install(TARGETS thunder_linalg DESTINATION lib)
install(DIRECTORY include/thunder DESTINATION include FILES_MATCHING PATTERN "*.hpp")

# Create tests
if(BUILD_THUNDER_TESTS)
  foreach(TEST_SOURCE ${TESTS})
    string(REPLACE ".cpp" "" TEST_TARGET ${TEST_SOURCE})
    string(REPLACE "test/" "" TEST_TARGET ${TEST_TARGET})
    add_executable(${TEST_TARGET} ${TEST_SOURCE})
    target_link_libraries(${TEST_TARGET} thunder_exception thunder_serializer thunder_storage thunder_tensor thunder_random thunder_linalg gtest gtest_main)
    add_test(${TEST_TARGET} ${TEST_TARGET})
  endforeach()
endif()
