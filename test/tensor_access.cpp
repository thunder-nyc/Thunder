/*
 * Copyright 2014 Xiang Zhang. All Rights Reserved.
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
 */

#include "thunder/tensor.hpp"

#include <memory>

#include "gtest/gtest.h"
#include "thunder/storage.hpp"

namespace thunder {
namespace {

// Written using static functions for simultaneously testing them
template < typename T >
void constructorTestCase(
    const T &tensor, typename T::dim_type dimension,
    const typename T::size_storage &size,
    const typename T::stride_storage &stride,
    typename T::size_type length, typename T::size_type offset,
    bool contiguity) {
  EXPECT_EQ(dimension, T::dimension(tensor));
  for (typename T::dim_type i = 0; i < dimension; ++i) {
    EXPECT_EQ(size[i], T::size(tensor, i));
    EXPECT_EQ(size[i], T::size(tensor)[i]);
    EXPECT_EQ(stride[i], T::stride(tensor, i));
    EXPECT_EQ(stride[i], T::stride(tensor)[i]);
  }
  EXPECT_EQ(length, T::length(tensor));
  EXPECT_EQ(offset, T::offset(tensor));
  EXPECT_EQ(contiguity, T::isContiguous(tensor));
  EXPECT_NE(nullptr, T::storage(tensor));
}
template < typename T >
void constructorTest() {
  // Default constructor create a size 1 tensor.
  T default_tensor;
  constructorTestCase(default_tensor, 1, {1}, {1}, 1, 0, true);

  // Create a 1-D tensor
  T one_dim_tensor(3);
  constructorTestCase(one_dim_tensor, 1, {3}, {1}, 3, 0, true);

  // Create a 2-D tensor
  T two_dim_tensor(3, 4);
  constructorTestCase(two_dim_tensor, 2, {3, 4}, {4, 1}, 12, 0, true);

  // Create a 3-D tensor
  T three_dim_tensor(3, 5, 4);
  constructorTestCase(three_dim_tensor, 3, {3, 5, 4}, {20, 4, 1}, 60, 0, true);

  // Create a 4-D tensor
  T four_dim_tensor(3, 5, 4, 7);
  constructorTestCase(
      four_dim_tensor, 4, {3, 5, 4, 7}, {140, 28, 7, 1}, 420, 0, true);

  // Create a 5-D tensor
  T five_dim_tensor({3, 5, 4, 7, 10});
  constructorTestCase(five_dim_tensor, 5, {3, 5, 4, 7, 10},
                      {1400, 280, 70, 10, 1}, 4200, 0, true);

  // Create a tensor from a storage
  T from_storage_tensor(::std::make_shared< typename T::storage_type >(37));
  constructorTestCase(from_storage_tensor, 1, {37}, {1}, 37, 0, true);

  // Create a tensor from a storage with offset
  T from_offset_storage_tensor(
      ::std::make_shared< typename T::storage_type >(37), 5);
  constructorTestCase(from_offset_storage_tensor, 1, {32}, {1}, 32, 5, true);

  // Create a tensor from a storage with size
  T from_size_storage_tensor(
      {5, 4}, ::std::make_shared< typename T::storage_type >(37));
  constructorTestCase(from_size_storage_tensor, 2, {5, 4}, {4, 1}, 20, 0, true);

  // Create a tensor from a storage with size and offset
  T from_size_offset_storage_tensor(
      {5, 4}, ::std::make_shared< typename T::storage_type >(37), 5);
  constructorTestCase(
      from_size_offset_storage_tensor, 2, {5, 4}, {4, 1}, 20, 5, true);

  // Create a tensor from size and stride
  T from_size_stride_tensor({5, 4}, {-1, 5});
  constructorTestCase(from_size_stride_tensor, 2, {5, 4}, {-1, 5}, 20, 4, false);
  EXPECT_EQ(20, from_size_stride_tensor.storage()->size());

  // Create a tensor from a storage with size and stride
  T from_size_stride_storage_tensor(
      {5, 4}, {5, 1}, ::std::make_shared< typename T::storage_type >(37));
  constructorTestCase(from_size_stride_storage_tensor,
                  2, {5, 4}, {5, 1}, 20, 0, false);

  // Create a tensor from a storage with size, stride and offset
  T from_size_stride_offset_storage_tensor(
      {5, 4}, {5, -1}, ::std::make_shared< typename T::storage_type >(37), 5);
  constructorTestCase(from_size_stride_offset_storage_tensor,
                  2, {5, 4}, {5, -1}, 20, 5, false);

  // Copy constructor
  T copy_tensor(from_size_stride_offset_storage_tensor);
  constructorTestCase(copy_tensor, 2, {5, 4}, {5, -1}, 20, 5, false);
  EXPECT_EQ(from_size_stride_offset_storage_tensor.storage(),
            copy_tensor.storage());
  EXPECT_EQ(from_size_stride_offset_storage_tensor.data(),
            copy_tensor.data());
}
TEST(TensorTest, constructorTest) {
  constructorTest< DoubleTensor >();
  constructorTest< FloatTensor >();
  constructorTest< DoubleComplexTensor >();
  constructorTest< FloatComplexTensor >();
}

template < typename T >
void accessTest() {
  // Create a tensor that is of normal stride
  T normal_tensor(3, 5, 4);

  // Test on evaluation operator
  normal_tensor(1, 3, 2) = 7;
  EXPECT_EQ(static_cast< typename T::value_type >(7), normal_tensor(1, 3, 2));
  EXPECT_EQ(normal_tensor.data(), &normal_tensor());
  EXPECT_EQ(normal_tensor.data() + 34, &normal_tensor.get(1, 3, 2));
  EXPECT_EQ(normal_tensor.data() + 34,
            &normal_tensor.get(
                typename T::size_storage({1, 3, 2})));
  EXPECT_TRUE(normal_tensor.partialContiguity(0, 1));

  // Test on sub-tensor selection operator
  T select_tensor = normal_tensor[1];
  EXPECT_EQ(2, select_tensor.dimension());
  EXPECT_EQ(5, select_tensor.size(0));
  EXPECT_EQ(4, select_tensor.size(1));
  EXPECT_EQ(4, select_tensor.stride(0));
  EXPECT_EQ(1, select_tensor.stride(1));
  EXPECT_EQ(normal_tensor.storage(), select_tensor.storage());
  EXPECT_EQ(20, select_tensor.offset());

  // Test on assignment operator
  T assign_tensor(7);
  assign_tensor = normal_tensor;
  EXPECT_EQ(normal_tensor.dimension(), assign_tensor.dimension());
  for (int i = 0; i < 3; ++i) {
    EXPECT_EQ(normal_tensor.size(i), assign_tensor.size(i));
    EXPECT_EQ(normal_tensor.stride(i), assign_tensor.stride(i));
  }
  EXPECT_EQ(normal_tensor.storage(), assign_tensor.storage());

  // Test on binary subtensor operator
  T normal_binary_subtensor = normal_tensor[{1, 2}];
  EXPECT_EQ(1, normal_binary_subtensor.dimension());
  EXPECT_EQ(4, normal_binary_subtensor.size(0));
  EXPECT_EQ(1, normal_binary_subtensor.stride(0));
  EXPECT_EQ(normal_tensor.storage(), normal_binary_subtensor.storage());
  EXPECT_EQ(normal_tensor.offset() + 20 + 8, normal_binary_subtensor.offset());
  EXPECT_TRUE(normal_binary_subtensor.isContiguous());

  // Test on multiple subtensor operator
  T normal_multiple_subtensor
      = normal_tensor[{{1, 2}, {2, 2}}];
  EXPECT_EQ(3, normal_multiple_subtensor.dimension());
  EXPECT_EQ(2, normal_multiple_subtensor.size(0));
  EXPECT_EQ(1, normal_multiple_subtensor.size(1));
  EXPECT_EQ(4, normal_multiple_subtensor.size(2));
  EXPECT_EQ(20, normal_multiple_subtensor.stride(0));
  EXPECT_EQ(4, normal_multiple_subtensor.stride(1));
  EXPECT_EQ(1, normal_multiple_subtensor.stride(2));
  EXPECT_EQ(normal_tensor.storage(), normal_multiple_subtensor.storage());
  EXPECT_EQ(normal_tensor.offset() + 20 + 8,
            normal_multiple_subtensor.offset());
  EXPECT_FALSE(normal_multiple_subtensor.isContiguous());

  // Create a tensor that is not of normal stride
  T stride_tensor({5, 4}, {-1, 5});

  // Test for partial contiguity
  EXPECT_TRUE(stride_tensor.partialContiguity(0, 0));
  EXPECT_FALSE(stride_tensor.partialContiguity(0, 1));
  EXPECT_FALSE(stride_tensor.partialContiguity(0, 2));

  // Test on evaluation operator
  for (int i = 0; i < 5; ++i) {
    for (int j = 0; j < 4; ++j) {
      stride_tensor(i, j) = i * 6 + j * 13 + 3;
      EXPECT_EQ(static_cast< typename T::value_type >(i * 6 + j * 13 + 3),
                stride_tensor.get(i, j));
      EXPECT_EQ(stride_tensor.data() - i + 5 * j, &stride_tensor.get(i, j));
    }
  }

  // Test on unitary subtensor operator
  for (int i = 0; i < 5; ++i) {
    T subtensor = stride_tensor[i];
    EXPECT_EQ(1, subtensor.dimension());
    EXPECT_EQ(4, subtensor.size(0));
    EXPECT_EQ(5, subtensor.stride(0));
    EXPECT_EQ(stride_tensor.offset() - i, subtensor.offset());
    EXPECT_EQ(stride_tensor.data() - i, subtensor.data());
    EXPECT_FALSE(subtensor.isContiguous());
  }

  // Test on binary subtensor operator
  T binary_subtensor = stride_tensor[{1, 3}];
  EXPECT_EQ(1, binary_subtensor.dimension());
  EXPECT_EQ(1, binary_subtensor.size(0));
  EXPECT_EQ(1, binary_subtensor.stride(0));
  EXPECT_EQ(stride_tensor.storage(), binary_subtensor.storage());
  EXPECT_EQ(stride_tensor.offset() - 1 + 3 * 5, binary_subtensor.offset());
  EXPECT_TRUE(binary_subtensor.isContiguous());

  // Test on multiple subtensor operator
  T multiple_subtensor = stride_tensor[{{1, 3}, {2, 2}}];
  EXPECT_EQ(2, multiple_subtensor.dimension());
  EXPECT_EQ(3, multiple_subtensor.size(0));
  EXPECT_EQ(1, multiple_subtensor.size(1));
  EXPECT_EQ(-1, multiple_subtensor.stride(0));
  EXPECT_EQ(5, multiple_subtensor.stride(1));
  EXPECT_EQ(stride_tensor.storage(), multiple_subtensor.storage());
  EXPECT_EQ(stride_tensor.offset() - 1 + 10, multiple_subtensor.offset());
  EXPECT_FALSE(multiple_subtensor.isContiguous());
}
TEST(TensorTest, accessTest) {
  accessTest< DoubleTensor >();
  accessTest< FloatTensor >();
  accessTest< DoubleComplexTensor >();
  accessTest< FloatComplexTensor >();
}

template < typename T >
void iteratorTest() {
  // Create a normal tensor
  T normal_tensor(3, 5, 4);

  // Subtensor iteration
  int i = 0;
  for (const T &t : normal_tensor) {
    EXPECT_EQ(2, t.dimension());
    EXPECT_EQ(5, t.size(0));
    EXPECT_EQ(4, t.size(1));
    EXPECT_EQ(4, t.stride(0));
    EXPECT_EQ(1, t.stride(1));
    EXPECT_EQ(normal_tensor.storage(), t.storage());
    EXPECT_EQ(i * 20, t.offset());
    EXPECT_TRUE(t.isContiguous());
    int j = 0;
    for (const T &s : t) {
      EXPECT_EQ(1, s.dimension());
      EXPECT_EQ(4, s.size(0));
      EXPECT_EQ(1, s.stride(0));
      EXPECT_EQ(normal_tensor.storage(), s.storage());
      EXPECT_EQ(i * 20 + j * 4, s.offset());
      EXPECT_TRUE(t.isContiguous());
      int k = 0;
      for (const T &r : s) {
        EXPECT_EQ(1, r.dimension());
        EXPECT_EQ(1, r.size(0));
        EXPECT_EQ(1, r.stride(0));
        EXPECT_EQ(normal_tensor.storage(), r.storage());
        EXPECT_EQ(i * 20 + j * 4 + k, r.offset());
        EXPECT_TRUE(r.isContiguous());
        int l = 0;
        for (const T &p : r) {
          EXPECT_EQ(1, p.dimension());
          EXPECT_EQ(1, p.size(0));
          EXPECT_EQ(1, p.stride(0));
          EXPECT_EQ(normal_tensor.storage(), p.storage());
          EXPECT_EQ(i * 20 + j * 4 + k, p.offset());
          EXPECT_TRUE(p.isContiguous());
          ++l;
        }
        EXPECT_EQ(1, l);
        ++k;
      }
      EXPECT_EQ(4, k);
      ++j;
    }
    EXPECT_EQ(5, j);
    ++i;
  }
  EXPECT_EQ(3, i);

  // Reference value iteration
  i = 0;
  for (auto begin = normal_tensor.reference_begin(),
           end = normal_tensor.reference_end();
       begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(i);
    EXPECT_EQ(static_cast< typename T::value_type >(i),
                    normal_tensor.data()[i]);
    ++i;
  }
  EXPECT_EQ(60, i);

  // Create a tensor that is not of normal stride
  T stride_tensor({5, 4}, {-1, 5});

  // Testing subtensor iterators
  i = 0;
  for (const T &t : stride_tensor) {
    EXPECT_EQ(1, t.dimension());
    EXPECT_EQ(4, t.size(0));
    EXPECT_EQ(5, t.stride(0));
    EXPECT_EQ(stride_tensor.storage(), t.storage());
    EXPECT_EQ(stride_tensor.offset() - i, t.offset());
    EXPECT_FALSE(t.isContiguous());
    int j = 0;
    for (const T &s : t) {
      EXPECT_EQ(1, s.dimension());
      EXPECT_EQ(1, s.size(0));
      EXPECT_EQ(1, s.stride(0));
      EXPECT_EQ(stride_tensor.storage(), s.storage());
      EXPECT_EQ(stride_tensor.offset() - i + 5 * j, s.offset());
      EXPECT_TRUE(s.isContiguous());
      int k = 0;
      for (const T &r : s) {
        EXPECT_EQ(1, r.dimension());
        EXPECT_EQ(1, r.size(0));
        EXPECT_EQ(1, r.stride(0));
        EXPECT_EQ(stride_tensor.storage(), r.storage());
        EXPECT_EQ(stride_tensor.offset() - i + 5 * j, r.offset());
        EXPECT_TRUE(r.isContiguous());
        ++k;
      }
      EXPECT_EQ(1, k);
      ++j;
    }
    EXPECT_EQ(4, j);
    ++i;
  }
  EXPECT_EQ(5, i);

  // Test reference iterators
  i = 0;
  for (auto begin = stride_tensor.reference_begin(),
           end = stride_tensor.reference_end();
       begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(i);
    EXPECT_EQ(static_cast< typename T::value_type >(i),
                    *(stride_tensor.data() - i / 4 + (i % 4) * 5));
    ++i;
  }
  EXPECT_EQ(20, i);
}
TEST(TensorTest, iteratorTest) {
  iteratorTest< DoubleTensor >();
  iteratorTest< FloatTensor >();
  iteratorTest< DoubleComplexTensor >();
  iteratorTest< FloatComplexTensor >();
}

}  // namespace
}  // namespace thunder
