/*
 * \copyright Copyright 2014 Xiang Zhang All Rights Reserved.
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
#include "thunder/storage/storage.hpp"

#include <complex>
#include <sstream>

#include "gtest/gtest.h"
#include "thunder/serializer.hpp"
#include "thunder/serializer/binary_protocol.hpp"
#include "thunder/serializer/serializer.hpp"
#include "thunder/serializer/static.hpp"
#include "thunder/serializer/text_protocol.hpp"

#include "thunder/serializer/binary_protocol-inl.hpp"
#include "thunder/serializer/serializer-inl.hpp"
#include "thunder/serializer/static-inl.hpp"
#include "thunder/serializer/text_protocol-inl.hpp"
#include "thunder/storage/storage-inl.hpp"

#define TEST_ALL_TYPES(FUNC)                             \
  TEST(StorageTest, FUNC) {                              \
    FUNC< double > ();                                   \
    FUNC< float > ();                                    \
    FUNC< int > ();                                      \
    FUNC< unsigned int > ();                             \
    FUNC< long > ();                                     \
    FUNC< unsigned long > ();                            \
    FUNC< long long > ();                                \
    FUNC< unsigned long long > ();                       \
    FUNC< char > ();                                     \
    FUNC< signed char > ();                              \
    FUNC< unsigned char > ();                            \
    FUNC< ::std::complex< double > > ();                 \
    FUNC< ::std::complex< float > > ();                  \
    FUNC< ::std::complex< int > > ();                    \
    FUNC< ::std::complex< unsigned int > > ();           \
    FUNC< ::std::complex< long > > ();                   \
    FUNC< ::std::complex< unsigned long > > ();          \
    FUNC< ::std::complex< long long > > ();              \
    FUNC< ::std::complex< unsigned long long > > ();     \
    FUNC< ::std::complex< char > > ();                   \
    FUNC< ::std::complex< signed char > > ();            \
    FUNC< ::std::complex< unsigned char > > ();          \
  }

template < typename T >
void constructorTest() {
  // Construct an empty storage
  thunder::Storage< T > default_storage;
  EXPECT_EQ(0, default_storage.size());
  EXPECT_EQ(nullptr, default_storage.data());

  // Construct an empty storage using explicit size
  thunder::Storage< T > size_empty_storage(0);
  EXPECT_EQ(0, size_empty_storage.size());
  EXPECT_EQ(nullptr, size_empty_storage.data());

  // Construct an empty storage from copy constructor
  thunder::Storage< T > copy_empty_storage(default_storage);
  EXPECT_EQ(0, copy_empty_storage.size());
  EXPECT_EQ(nullptr, copy_empty_storage.data());

  // Construct an storage with some size
  thunder::Storage< T > size_storage(5);
  int size_storage_val = 0;
  for (typename thunder::Storage< T >::size_type i = 0; i < 5; ++i) {
    size_storage[i] = static_cast< T >(size_storage_val++);
  }
  EXPECT_EQ(5, size_storage.size());
  EXPECT_NE(nullptr, size_storage.data());

  // Construct an storage with some size and a default value
  thunder::Storage< T > size_value_storage(5, (T)3);
  EXPECT_EQ(5, size_storage.size());
  EXPECT_NE(nullptr, size_value_storage.data());
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ((T)3, size_value_storage.data()[i]);
  }

  // Copy construct a storage
  thunder::Storage< T > copy_storage(size_storage);
  EXPECT_EQ(5, copy_storage.size());
  EXPECT_NE(nullptr, copy_storage.data());
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(size_storage.data()[i], copy_storage.data()[i]);
  }

  // Initialzation list constructor
  thunder::Storage< T > init_storage({3, 4, 5, 6, 7});
  EXPECT_EQ(5, init_storage.size());
  EXPECT_NE(nullptr, init_storage.data());
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(static_cast< T >(i + 3), init_storage.data()[i]);
  }

  // Construct an storage using external data
  thunder::Storage< T > data_storage(
      init_storage.data(), init_storage.size());
  EXPECT_EQ(5, data_storage.size());
  EXPECT_NE(nullptr, data_storage.data());
  EXPECT_EQ(init_storage.data(), data_storage.data());
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(static_cast< T >(i + 3), data_storage.data()[i]);
  }

  // Construct a storage using external shared data
  thunder::Storage< T > shared_storage(
      init_storage.shared(), init_storage.size());
  EXPECT_EQ(5, shared_storage.size());
  EXPECT_NE(nullptr, shared_storage.data());
  EXPECT_EQ(init_storage.data(), shared_storage.data());
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(static_cast< T >(i + 3), shared_storage.data()[i]);
  }
}
TEST_ALL_TYPES(constructorTest);

template < typename T >
void assignmentTest() {
  // Create a storage of size 5 and assign values to it
  thunder::Storage< T > storage(5);
  for (int i = 0; i < 5; ++i) {
    storage.data()[i] = static_cast< T >(i + 3);
  }

  // Assign the storage
  thunder::Storage< T > assign_storage(10);
  assign_storage = storage;
  EXPECT_EQ(5, assign_storage.size());
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(storage.data()[i], assign_storage.data()[i]);
  }

  // Assign to self
  assign_storage = assign_storage;
  EXPECT_EQ(5, assign_storage.size());
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(storage.data()[i], assign_storage.data()[i]);
  }
}
TEST_ALL_TYPES(assignmentTest);

template < typename T >
void indexAndIteratorTest() {
  // Create a storage of size 5 and assign values to it
  thunder::Storage< T > storage(5);
  for (int i = 0; i < 5; ++i) {
    storage[i] = static_cast< T >(i + 3);
    EXPECT_EQ(static_cast< T >(i + 3), storage[i]);
  }
  int i = 0;
  for (const T& val : storage) {
    EXPECT_EQ(static_cast< T >(i + 3), val);
    ++i;
  }
}
TEST_ALL_TYPES(indexAndIteratorTest);

template < typename T >
void copyTest() {
  // Create an int storage of size 5 and assign values to it
  thunder::Storage<int> double_storage(5);
  for (int i = 0; i < 5; ++i) {
    double_storage[i] = static_cast< double >(i + 3);
  }

  // Copy to self
  double_storage.copy(double_storage);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(static_cast< double >(i + 3), double_storage[i]);
  }

  // Create an empty storage of type T
  thunder::Storage< T > storage;
  storage.copy(double_storage);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ(static_cast< T >(i + 3), storage[i]);
  }
}
TEST_ALL_TYPES(copyTest);

template < typename T >
void resizeTest() {
  // Create a storage of size 10
  thunder::Storage< T > storage(10);

  // Resize to 5 with default value
  storage.resize(5, (T)3);
  for (int i = 0; i < 5; ++i) {
    EXPECT_EQ((T)3, storage[i]);
  }
}
TEST_ALL_TYPES(resizeTest);

template < typename T >
void allocatorTest() {
  // Create an allocator
  ::std::allocator< T > alloc;

  thunder::Storage< T > storage(10);
  EXPECT_TRUE(alloc == storage.allocator());
}

template < typename T >
void serializeTest() {
  // Create a storage of some size
  thunder::Storage< T > s1(17);
  for (int i = 0; i < 17; ++i) {
    s1[i] = i + 4;
  }

  ::thunder::StringBinarySerializer a1;
  thunder::Storage< T > s2;
  a1.save(s1);
  a1.load(&s2);
  EXPECT_EQ(s1.size(), s2.size());
  for (int i = 0; i < s1.size(); ++i) {
    EXPECT_EQ(s1[i], s2[i]);
  }

  ::thunder::StringTextSerializer a2;
  thunder::Storage< T > s3;
  a2.save(s1);
  a2.load(&s3);
  EXPECT_EQ(s1.size(), s3.size());
  for (int i = 0; i < s1.size(); ++i) {
    EXPECT_EQ(s1[i], s3[i]);
  }
}
TEST_ALL_TYPES(serializeTest);

template < typename T >
void viewTest() {
  thunder::Storage< ::std::complex< T > > s1(10);
  for (int i = 0; i < s1.size(); ++i) {
    s1[i] = ::std::complex< T >(static_cast< T >(i), static_cast< T >(i - 3));
  }
  thunder::Storage< T > s2 = s1.template view< thunder::Storage< T > >();

  EXPECT_EQ(s1.data(), reinterpret_cast< ::std::complex< T >* >(s2.data()));
  EXPECT_EQ(2 * s1.size(), s2.size());
  for (int i = 0; i < s1.size(); ++i) {
    EXPECT_EQ(::std::real(s1[i]), s2[i * 2]);
    EXPECT_EQ(::std::imag(s1[i]), s2[i * 2 + 1]);
  }
}
TEST(StorageTest, viewTest) {
  viewTest< double >();
  viewTest< float >();
}

#undef TEST_ALL_TYPES
