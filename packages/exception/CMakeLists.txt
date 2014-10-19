# Get all the include and source files
file(GLOB_RECURSE HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "include/thunder/*.hpp")
file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "src/*.cpp")

# Create the library
add_library(thunder_exception ${HEADERS} ${SOURCES})
target_include_directories(thunder_exception PUBLIC "include")

# Create installation
install(TARGETS thunder_exception DESTINATION lib)
install(DIRECTORY include/thunder DESTINATION include FILES_MATCHING PATTERN "*.hpp")
