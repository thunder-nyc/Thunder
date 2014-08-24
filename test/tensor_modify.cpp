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
void modifyTest() {
  // Create 2 tensors and set/copy between them
  T tensor1(5, 3, 4);
  T tensor2(7, 9, 2);
  T tensor3(2, 1, 2);
  FloatTensor tensor4(7, 9, 2);
  T tensor5({7, 9, 2}, {-1, 10, 17});

  T::set(&tensor1, tensor2);
  EXPECT_EQ(tensor2.storage(), tensor1.storage());
  EXPECT_EQ(tensor2.offset(), tensor1.offset());
  EXPECT_EQ(5, tensor1.size(0));
  EXPECT_EQ(3, tensor1.size(1));
  EXPECT_EQ(4, tensor1.size(2));
  EXPECT_EQ(12, tensor1.stride(0));
  EXPECT_EQ(4, tensor1.stride(1));
  EXPECT_EQ(1, tensor1.stride(2));

  // Test on resizeAs
  T::resizeAs(&tensor1, tensor2);
  EXPECT_TRUE(T::isSameSizeAs(tensor1, tensor2));
  EXPECT_NE(tensor1.storage(), tensor2.storage());
  T::resizeAs(&tensor3, tensor4);
  EXPECT_TRUE(T::isSameSizeAs(tensor3, tensor4));

  // Test on copy
  tensor2(3) = 17;
  T::copy(&tensor1, tensor2);
  EXPECT_FLOAT_EQ(17, tensor1(3));
  T::copy(&tensor5, tensor2);
  EXPECT_FLOAT_EQ(17, tensor5(3));
  tensor4(3, 2) = 19;
  T::copy(&tensor1, tensor4);
  EXPECT_FLOAT_EQ(19, tensor1(3, 2));
  T::copy(&tensor5, tensor4);
  EXPECT_FLOAT_EQ(19, tensor5(3, 2));

  // Test on set with storage
  typename T::storage_pointer s
      = std::make_shared< typename T::storage_type >(1000);
  T::set(&tensor1, s, 103);
  EXPECT_EQ(s, tensor1.storage());
  typename T::size_storage sz({5, 8, 10});
  T::set(&tensor1, sz, s, 103);
  EXPECT_EQ(s, tensor1.storage());
  EXPECT_EQ(5, tensor1.size(0));
  EXPECT_EQ(8, tensor1.size(1));
  EXPECT_EQ(10, tensor1.size(2));
  typename T::stride_storage st({80, 9, 2});
  T::set(&tensor1, sz, st, s, 103);
  EXPECT_EQ(s, tensor1.storage());
  EXPECT_EQ(5, tensor1.size(0));
  EXPECT_EQ(8, tensor1.size(1));
  EXPECT_EQ(10, tensor1.size(2));
  EXPECT_EQ(80, tensor1.stride(0));
  EXPECT_EQ(9, tensor1.stride(1));
  EXPECT_EQ(2, tensor1.stride(2));
  EXPECT_FALSE(tensor1.isContiguous());

  // Make tensor 1 contiguous
  T::contiguous(&tensor1);
  EXPECT_NE(s, tensor1.storage());
  EXPECT_EQ(5, tensor1.size(0));
  EXPECT_EQ(8, tensor1.size(1));
  EXPECT_EQ(10, tensor1.size(2));
  EXPECT_TRUE(tensor1.isContiguous());

  // Squeeze tensor 3
  tensor3.resize(3, 1, 2);
  T::squeeze(&tensor3);
  EXPECT_EQ(2, tensor3.dimension());
  EXPECT_EQ(3, tensor3.size(0));
  EXPECT_EQ(2, tensor3.size(1));
  EXPECT_TRUE(tensor3.isContiguous());

  // Make a subtensor of tensor 3
  T tensor6 = tensor3[1];
  // 3 because storage() call create a temporary
  EXPECT_EQ(3, tensor3.storage().use_count());
  EXPECT_EQ(3, tensor6.storage().use_count());
  T::unique(&tensor6);
  // 2 because storage() call create a temporary
  EXPECT_EQ(2, tensor3.storage().use_count());
  EXPECT_EQ(2, tensor6.storage().use_count());
}

TEST(TensorTest, modifyTest) {
  modifyTest< DoubleTensor >();
  modifyTest< FloatTensor >();
  modifyTest< Tensor< Storage< int > > > ();
}

template < typename T >
void transformTest() {
  T t1(10, 20, 7);
  FloatTensor t2(3, 5);

  T t1_viewas = T::viewAs(t1, t2, 10);
  EXPECT_EQ(t2.dimension(), t1_viewas.dimension());
  EXPECT_EQ(t2.size(0), t1_viewas.size(0));
  EXPECT_EQ(t2.size(1), t1_viewas.size(1));
  EXPECT_EQ(10, t1_viewas.offset());
  EXPECT_EQ(t1.storage(), t1_viewas.storage());

  T t1_viewas_stride = T::viewAs(
      t1, t2, typename T::stride_storage({1, 7}), 10);
  EXPECT_EQ(t2.dimension(), t1_viewas_stride.dimension());
  EXPECT_EQ(t2.size(0), t1_viewas_stride.size(0));
  EXPECT_EQ(t2.size(1), t1_viewas_stride.size(1));
  EXPECT_EQ(10, t1_viewas_stride.offset());
  EXPECT_EQ(t1.storage(), t1_viewas_stride.storage());
}

TEST(TensorTest, transformTest) {
  transformTest< DoubleTensor >();
  transformTest< FloatTensor >();
}

}  // namespace
}  // namespace thunder
