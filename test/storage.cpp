#include "thunder/storage.hpp"

#include "gtest/gtest.h"

template <typename T>
void ConstructorTest() {
  // Construct an empty storage
  thunder::Storage<T> default_storage;
  EXPECT_EQ(0, default_storage.Size()) << "Storage size should be 0.";
  EXPECT_EQ(nullptr, default_storage.Data())
      << "Storage data should be nullptr.";

  // Construct an empty storage using explicit size
  thunder::Storage<T> size_empty_storage(0);
  EXPECT_EQ(0, size_empty_storage.Size()) << "Storage size should be 0.";
  EXPECT_EQ(nullptr, size_empty_storage.Data())
      << "Storage data should be nullptr.";

  // Construct an empty storage from copy constructor
  thunder::Storage<T> copy_empty_storage(default_storage);
  EXPECT_EQ(0, copy_empty_storage.Size()) << "Storage size should be 0.";
  EXPECT_EQ(nullptr, copy_empty_storage.Data())
      << "Storage data should be nullptr.";
}

TEST(StorageTest, ConstructorTest) {
  ConstructorTest<double>();
}
