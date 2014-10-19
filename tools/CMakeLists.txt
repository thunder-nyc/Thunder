# Add gtest if test is enabled
if (BUILD_THUNDER_TESTS)
  add_subdirectory(gtest-1.7.0)
endif ()
