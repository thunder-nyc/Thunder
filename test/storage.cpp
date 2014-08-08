/*
 * \copyright Copyright 2014 Xiang Zhang. All Rights Reserved.
 * \license @{
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * @}
 */

#include "thunder/storage.hpp"

#include <stdio.h>

#include "gtest/gtest.h"

#define TEST_ALL_TYPES(FUNC)         \
  TEST(StorageTest, FUNC) {          \
    FUNC<double> ();                 \
    FUNC<float> ();                  \
    FUNC<int> ();                    \
    FUNC<unsigned int> ();           \
    FUNC<long> ();                   \
    FUNC<unsigned long> ();          \
    FUNC<long long> ();              \
    FUNC<unsigned long long> ();     \
    FUNC<char> ();                   \
    FUNC<signed char> ();            \
    FUNC<unsigned char> ();          \
  }

template <typename T>
void ConstructorTest() {
  // Construct an empty storage
  thunder::Storage<T> default_storage;
  EXPECT_EQ(0, default_storage.Size());
  EXPECT_EQ(nullptr, default_storage.Data());

  // Construct an empty storage using explicit size
  thunder::Storage<T> size_empty_storage(0);
  EXPECT_EQ(0, size_empty_storage.Size());
  EXPECT_EQ(nullptr, size_empty_storage.Data());

  // Construct an empty storage from copy constructor
  thunder::Storage<T> copy_empty_storage(default_storage);
  EXPECT_EQ(0, copy_empty_storage.Size());
  EXPECT_EQ(nullptr, copy_empty_storage.Data());

  // Construct an storage with some size
  thunder::Storage<T> size_storage(5);
  EXPECT_EQ(5, size_storage.Size());
  EXPECT_NE(nullptr, size_storage.Data());

  // Construct an storage with some size and a default value
  thunder::Storage<T> size_value_storage(5, (T)3);
  EXPECT_EQ(5, size_storage.Size());
  EXPECT_NE(nullptr, size_value_storage.Data());
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(3, size_value_storage.Data()[i]);
  }

  // Copy construct a storage
  thunder::Storage<T> copy_storage(size_storage);
  EXPECT_EQ(5, copy_storage.Size());
  EXPECT_NE(nullptr, copy_storage.Data());
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(size_storage.Data()[i], copy_storage.Data()[i]);
  }
}
TEST_ALL_TYPES(ConstructorTest);

template <typename T>
void AssignmentTest() {
  // Create a storage of size 5 and assign values to it
  thunder::Storage<T> storage(5);
  for (int i = 0; i < 5; ++i) {
    storage.Data()[i] = static_cast<T>(i + 3);
  }

  // Assign the storage
  thunder::Storage<T> assign_storage(10);
  assign_storage = storage;
  EXPECT_EQ(5, assign_storage.Size());
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(storage.Data()[i], assign_storage.Data()[i]);
  }

  // Assign to self
  assign_storage = assign_storage;
  EXPECT_EQ(5, assign_storage.Size());
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(storage.Data()[i], assign_storage.Data()[i]);
  }
}
TEST_ALL_TYPES(AssignmentTest);

template <typename T>
void IndexAndIteratorTest() {
  // Create a storage of size 5 and assign values to it
  thunder::Storage<T> storage(5);
  for (int i = 0; i < 5; ++i) {
    storage[i] = static_cast<T>(i + 3);
    EXPECT_EQ(static_cast<T>(i + 3), storage[i]);
  }
  int i = 0;
  for (const T& val : storage) {
    EXPECT_EQ(static_cast<T>(i + 3), val);
    ++i;
  }
}
TEST_ALL_TYPES(IndexAndIteratorTest);

template <typename T>
void CopyTest() {
  // Create an int storage of size 5 and assign values to it
  thunder::Storage<int> double_storage(5);
  for (int i = 0; i < 5; ++i) {
    double_storage[i] = static_cast<double>(i + 3);
  }

  // Copy to self
  double_storage.Copy(double_storage);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(static_cast<double>(i + 3), double_storage[i]);
  }

  // Create an empty storage of type T
  thunder::Storage<T> storage;
  storage.Copy(double_storage);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(static_cast<T>(i + 3), storage[i]);
  }
}
TEST_ALL_TYPES(CopyTest);

template <typename T>
void ResizeTest() {
  // Create a storage of size 10
  thunder::Storage<T> storage(10);

  // Resize to 5 with default value
  storage.Resize(5, (T)3);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ((T)3, storage[i]);
  }
}
TEST_ALL_TYPES(ResizeTest);

#undef TEST_ALL_TYPES
