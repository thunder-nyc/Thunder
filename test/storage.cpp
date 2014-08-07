#include "thunder/storage.hpp"

#include "gtest/gtest.h"

template <typename T>
void EmptyTest() {
  // Construct an empty storage
  thunder::Storage<T> default_storage;
  EXPECT_EQ(0, default_storage.Size()) << "Storage size should be 0.";
  EXPECT_EQ(nullptr, default_storage.Data())
      << "Storage data should be nullptr.";

  // Construct an empty storage using explicit size
  thunder::Storage<T> size_storage(0);
  EXPECT_EQ(0, size_storage.Size()) << "Storage size should be 0.";
  EXPECT_EQ(nullptr, size_storage.Data())
      << "Storage data should be nullptr.";

  // Construct an empty storage from copy constructor
  thunder::Storage<T> copy_storage(size_storage);
  EXPECT_EQ(0, copy_storage.Size()) << "Storage size should be 0.";
  EXPECT_EQ(nullptr, copy_storage.Data())
      << "Storage data should be nullptr.";

  // Empty storage from assignment operator
  thunder::Storage<T> assign_storage(5);
  assign_storage = size_storage;
  EXPECT_EQ(0, assign_storage.Size()) << "Storage size should be 0.";
  EXPECT_EQ(nullptr, assign_storage.Data())
      << "Storage data should be nullptr.";
}

TEST(StorageTest, DoubleTest) {
  EmptyTest<double>();
}
