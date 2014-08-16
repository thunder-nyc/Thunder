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

template < typename T >
void constructorTest(
    const T &tensor, typename T::dim_type dimension,
    const typename T::size_storage &size,
    const typename T::stride_storage &stride,
    typename T::size_type count, typename T::size_type offset,
    bool contiguity) {
  EXPECT_EQ(dimension, tensor.dimension());
  for(typename T::dim_type i = 0; i < dimension; ++i) {
    EXPECT_EQ(size[i], tensor.size(i));
    EXPECT_EQ(size[i], tensor.size()[i]);
    EXPECT_EQ(stride[i], tensor.stride(i));
    EXPECT_EQ(stride[i], tensor.stride()[i]);
  }
  EXPECT_EQ(count, tensor.count());
  EXPECT_EQ(offset, tensor.offset());
  EXPECT_EQ(contiguity, tensor.isContiguous());
  EXPECT_NE(nullptr, tensor.storage());
}

TEST(TensorTest, constructorTest) {
  // Default constructor create a size 1 tensor.
  Tensor< DoubleStorage > default_tensor;
  constructorTest(default_tensor, 1, {1}, {1}, 1, 0, true);

  // Create a 1-D tensor
  Tensor< DoubleStorage > one_dim_tensor(3);
  constructorTest(one_dim_tensor, 1, {3}, {1}, 3, 0, true);

  // Create a 2-D tensor
  Tensor< DoubleStorage > two_dim_tensor(3, 4);
  constructorTest(two_dim_tensor, 2, {3, 4}, {4, 1}, 12, 0, true);

  // Create a 3-D tensor
  Tensor< DoubleStorage > three_dim_tensor(3, 5, 4);
  constructorTest(three_dim_tensor, 3, {3, 5, 4}, {20, 4, 1}, 60, 0, true);

  // Create a 4-D tensor
  Tensor< DoubleStorage > four_dim_tensor(3, 5, 4, 7);
  constructorTest(
      four_dim_tensor, 4, {3, 5, 4, 7}, {140, 28, 7, 1}, 420, 0, true);

  // Create a 5-D tensor
  Tensor< DoubleStorage > five_dim_tensor({3, 5, 4, 7, 10});
  constructorTest(five_dim_tensor, 5, {3, 5, 4, 7, 10}, {1400, 280, 70, 10, 1},
                  4200, 0, true);

  // Create a tensor from a storage
  Tensor< DoubleStorage > from_storage_tensor(
      ::std::make_shared< DoubleStorage >(37));
  constructorTest(from_storage_tensor, 1, {37}, {1}, 37, 0, true);

  // Create a tensor from a storage with offset
  Tensor< DoubleStorage > from_offset_storage_tensor(
      ::std::make_shared< DoubleStorage >(37), 5);
  constructorTest(from_offset_storage_tensor, 1, {32}, {1}, 32, 5, true);

  // Create a tensor from a storage with size
  Tensor< DoubleStorage > from_size_storage_tensor(
      {5, 4}, ::std::make_shared< DoubleStorage >(37));
  constructorTest(from_size_storage_tensor, 2, {5, 4}, {4, 1}, 20, 0, true);

  // Create a tensor from a storage with size and offset
  Tensor< DoubleStorage > from_size_offset_storage_tensor(
      {5, 4}, ::std::make_shared< DoubleStorage >(37), 5);
  constructorTest(
      from_size_offset_storage_tensor, 2, {5, 4}, {4, 1}, 20, 5, true);

  // Create a tensor from size and stride
  Tensor< DoubleStorage > from_size_stride_tensor({5, 4}, {-1, 5});
  constructorTest(from_size_stride_tensor, 2, {5, 4}, {-1, 5}, 20, 4, false);
  EXPECT_EQ(20, from_size_stride_tensor.storage()->size());

  // Create a tensor from a storage with size and stride
  Tensor< DoubleStorage > from_size_stride_storage_tensor(
      {5, 4}, {5, 1}, ::std::make_shared< DoubleStorage >(37));
  constructorTest(from_size_stride_storage_tensor,
                  2, {5, 4}, {5, 1}, 20, 0, false);

  // Create a tensor from a storage with size, stride and offset
  Tensor< DoubleStorage > from_size_stride_offset_storage_tensor(
      {5, 4}, {5, -1}, ::std::make_shared< DoubleStorage >(37), 5);
  constructorTest(from_size_stride_offset_storage_tensor,
                  2, {5, 4}, {5, -1}, 20, 5, false);

  // Copy constructor
  Tensor< DoubleStorage > copy_tensor(from_size_stride_offset_storage_tensor);
  constructorTest(copy_tensor,2, {5, 4}, {5, -1}, 20, 5, false);
  EXPECT_EQ(from_size_stride_offset_storage_tensor.storage(),
            copy_tensor.storage());
  EXPECT_EQ(from_size_stride_offset_storage_tensor.data(),
            copy_tensor.data());
}

}  // namespace
}  // namespace thunder
