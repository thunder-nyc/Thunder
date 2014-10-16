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
  modifyTest< DoubleComplexTensor >();
  modifyTest< FloatComplexTensor >();
}

template < typename T >
void viewAsTest() {
  T t1(10, 20, 7);
  int t1_val = 0;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++);
  }

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

TEST(TensorTest, viewAsTest) {
  viewAsTest< DoubleTensor >();
  viewAsTest< FloatTensor >();
  viewAsTest< DoubleComplexTensor >();
  viewAsTest< FloatComplexTensor >();
}

template < typename T >
void extractTest() {
  T t1(10, 20, 7);
  int t1_val = 0;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++);
  }

  FloatTensor t2(10, 20, 7);
  for (FloatTensor::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = 0;
  }
  t2(1, 9, 3) = 1;
  t2(5, 4, 4) = 1;
  t2(9, 2, 8) = 1;
  T t1_extracted = T::extract(t1, t2);
  EXPECT_EQ(1, t1_extracted.dimension());
  EXPECT_EQ(3, t1_extracted.size(0));
  EXPECT_EQ(t1(1, 9, 3), t1_extracted(0));
  EXPECT_EQ(t1(5, 4, 4), t1_extracted(1));
  EXPECT_EQ(t1(9, 2, 8), t1_extracted(2));

  FloatTensor t3({10, 20}, {40, 2});
  for (FloatTensor::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    *begin = 0;
  }
  t3(2, 4) = 1;
  t3(9, 7) = 1;
  T t1_extracted_stride = T::extract(t1, t3);
  EXPECT_EQ(2, t1_extracted_stride.dimension());
  EXPECT_EQ(2, t1_extracted_stride.size(0));
  EXPECT_EQ(7, t1_extracted_stride.size(1));
  for (typename T::dim_type i = 0; i < 7; ++i) {
    EXPECT_EQ(t1(2, 4, i), t1_extracted_stride(0, i));
    EXPECT_EQ(t1(9, 7, i), t1_extracted_stride(1, i));
  }
}

TEST(TensorTest, extractTest) {
  extractTest< DoubleTensor >();
  extractTest< FloatTensor >();
  extractTest< DoubleComplexTensor >();
  extractTest< FloatComplexTensor >();
}

template < typename T >
void shuffleTest() {
  T t1(10, 20, 7);
  int t1_val = 0;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++);
  }

  FloatTensor t2(3);
  t2(0) = 2;
  t2(1) = 7;
  t2(2) = 5;
  T t3 = T::shuffle(t1, t2);
  EXPECT_EQ(1, t3.dimension());
  EXPECT_EQ(1, t3.size(0));
  EXPECT_EQ(t1(2, 7, 5), t3(0));

  FloatTensor t4(10, 3);
  for (int i = 0; i < 10; ++i) {
    t4(i, 0) = i;
    t4(i, 1) = 10 + i;
    t4(i, 2) = i / 2;
  }
  T t5 = T::shuffle(t1, t4);
  EXPECT_EQ(1, t5.dimension());
  EXPECT_EQ(10, t5.size(0));
  for (int i = 0; i < 10; ++i) {
    EXPECT_EQ(t1(i, 10 + i, i / 2), t5(i));
  }
}

TEST(TensorTest, shuffleTest) {
  shuffleTest< DoubleTensor >();
  shuffleTest< FloatTensor >();
  shuffleTest< DoubleComplexTensor >();
  shuffleTest< FloatComplexTensor >();
}

template< typename T >
void viewTest() {
  T t1(10, 20, 7);
  int t1_val = 0;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++);
  }

  T t1_viewed_1 = T::view(t1, 90);
  EXPECT_EQ(1, t1_viewed_1.dimension());
  EXPECT_EQ(90, t1_viewed_1.size(0));
  EXPECT_EQ(t1.storage(), t1_viewed_1.storage());
  EXPECT_EQ(0, t1_viewed_1.offset());

  T t1_viewed_2 = T::view(t1, 90, 3);
  EXPECT_EQ(2, t1_viewed_2.dimension());
  EXPECT_EQ(90, t1_viewed_2.size(0));
  EXPECT_EQ(3, t1_viewed_2.size(1));
  EXPECT_EQ(t1.storage(), t1_viewed_2.storage());
  EXPECT_EQ(0, t1_viewed_2.offset());

  T t1_viewed_3 = T::view(t1, 90, 3, 2);
  EXPECT_EQ(3, t1_viewed_3.dimension());
  EXPECT_EQ(90, t1_viewed_3.size(0));
  EXPECT_EQ(3, t1_viewed_3.size(1));
  EXPECT_EQ(2, t1_viewed_3.size(2));
  EXPECT_EQ(t1.storage(), t1_viewed_3.storage());
  EXPECT_EQ(0, t1_viewed_3.offset());

  T t1_viewed_4 = T::view(t1, 90, 3, 2, 2);
  EXPECT_EQ(4, t1_viewed_4.dimension());
  EXPECT_EQ(90, t1_viewed_4.size(0));
  EXPECT_EQ(3, t1_viewed_4.size(1));
  EXPECT_EQ(2, t1_viewed_4.size(2));
  EXPECT_EQ(2, t1_viewed_4.size(3));
  EXPECT_EQ(t1.storage(), t1_viewed_4.storage());
  EXPECT_EQ(0, t1_viewed_4.offset());

  T t1_viewed_size = T::view(t1, {90, 3, 2}, 10);
  EXPECT_EQ(3, t1_viewed_size.dimension());
  EXPECT_EQ(90, t1_viewed_size.size(0));
  EXPECT_EQ(3, t1_viewed_size.size(1));
  EXPECT_EQ(2, t1_viewed_size.size(2));
  EXPECT_EQ(t1.storage(), t1_viewed_size.storage());
  EXPECT_EQ(10, t1_viewed_size.offset());

  T t1_viewed_stride = T::view(t1, {90, 3, 2}, {2, 7, -1}, 10);
  EXPECT_EQ(3, t1_viewed_stride.dimension());
  EXPECT_EQ(90, t1_viewed_stride.size(0));
  EXPECT_EQ(3, t1_viewed_stride.size(1));
  EXPECT_EQ(2, t1_viewed_stride.size(2));
  EXPECT_EQ(2, t1_viewed_stride.stride(0));
  EXPECT_EQ(7, t1_viewed_stride.stride(1));
  EXPECT_EQ(-1, t1_viewed_stride.stride(2));
  EXPECT_EQ(t1.storage(), t1_viewed_stride.storage());
  EXPECT_EQ(10, t1_viewed_stride.offset());
}

TEST(TensorTest, viewTest) {
  viewTest< DoubleTensor >();
  viewTest< FloatTensor >();
  viewTest< DoubleComplexTensor >();
  viewTest< FloatComplexTensor >();
}

template< typename T >
void reshapeTest() {
  T t1(10, 20, 7);
  int t1_val = 0;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++);
  }

  T t1_reshaped_1 = T::reshape(t1, 1400);
  EXPECT_EQ(1, t1_reshaped_1.dimension());
  EXPECT_EQ(1400, t1_reshaped_1.size(0));
  EXPECT_EQ(t1.storage(), t1_reshaped_1.storage());

  T t1_reshaped_2 = T::reshape(t1, 700, 2);
  EXPECT_EQ(2, t1_reshaped_2.dimension());
  EXPECT_EQ(700, t1_reshaped_2.size(0));
  EXPECT_EQ(2, t1_reshaped_2.size(1));
  EXPECT_EQ(t1.storage(), t1_reshaped_2.storage());

  T t1_reshaped_3 = T::reshape(t1, 100, 7, 2);
  EXPECT_EQ(3, t1_reshaped_3.dimension());
  EXPECT_EQ(100, t1_reshaped_3.size(0));
  EXPECT_EQ(7, t1_reshaped_3.size(1));
  EXPECT_EQ(2, t1_reshaped_3.size(2));
  EXPECT_EQ(t1.storage(), t1_reshaped_3.storage());

  T t1_reshaped_4 = T::reshape(t1, 10, 10, 7, 2);
  EXPECT_EQ(4, t1_reshaped_4.dimension());
  EXPECT_EQ(10, t1_reshaped_4.size(0));
  EXPECT_EQ(10, t1_reshaped_4.size(1));
  EXPECT_EQ(7, t1_reshaped_4.size(2));
  EXPECT_EQ(2, t1_reshaped_4.size(3));
  EXPECT_EQ(t1.storage(), t1_reshaped_4.storage());

  T t1_reshaped_5 = T::reshape(t1, {10, 2, 5, 7, 2});
  EXPECT_EQ(5, t1_reshaped_5.dimension());
  EXPECT_EQ(10, t1_reshaped_5.size(0));
  EXPECT_EQ(2, t1_reshaped_5.size(1));
  EXPECT_EQ(5, t1_reshaped_5.size(2));
  EXPECT_EQ(7, t1_reshaped_5.size(3));
  EXPECT_EQ(2, t1_reshaped_5.size(4));
  EXPECT_EQ(t1.storage(), t1_reshaped_5.storage());
}

TEST(TensorTest, reshapeTest) {
  reshapeTest< DoubleTensor >();
  reshapeTest< FloatTensor >();
  reshapeTest< DoubleComplexTensor >();
  reshapeTest< FloatComplexTensor >();
}

template < typename T >
void transformTest() {
  T t1(10, 20, 7);
  int t1_val = 0;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++);
  }

  T t1_narrowed = T::narrow(t1, 1, 2, 5);
  EXPECT_EQ(3, t1_narrowed.dimension());
  EXPECT_EQ(10, t1_narrowed.size(0));
  EXPECT_EQ(5, t1_narrowed.size(1));
  EXPECT_EQ(7, t1_narrowed.size(2));
  EXPECT_EQ(t1.storage(), t1_narrowed.storage());
  EXPECT_EQ(t1.offset() + 2 * 7, t1_narrowed.offset());
  EXPECT_FALSE(t1_narrowed.isContiguous());
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 5; ++j) {
      for (int k = 0; k < 7; ++k) {
        EXPECT_EQ(t1(i, j + 2, k), t1_narrowed(i, j, k));
      }
    }
  }

  T t1_selected = T::select(t1, 1, 2);
  EXPECT_EQ(2, t1_selected.dimension());
  EXPECT_EQ(10, t1_selected.size(0));
  EXPECT_EQ(7, t1_selected.size(1));
  EXPECT_EQ(t1.storage(), t1_selected.storage());
  EXPECT_EQ(t1.offset() + 2 * 7, t1_selected.offset());
  EXPECT_FALSE(t1_selected.isContiguous());
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 7; ++j) {
      EXPECT_EQ(t1(i, 2, j), t1_selected(i, j));
    }
  }

  T t1_transposed = T::transpose(t1);
  EXPECT_EQ(3, t1_transposed.dimension());
  EXPECT_EQ(20, t1_transposed.size(0));
  EXPECT_EQ(10, t1_transposed.size(1));
  EXPECT_EQ(7, t1_transposed.size(2));
  EXPECT_EQ(t1.storage(), t1_transposed.storage());
  EXPECT_FALSE(t1_transposed.isContiguous());
  for (int i = 0; i < 20; ++i) {
    for (int j = 0; j < 10; ++j) {
      for (int k = 0; k < 7; ++k) {
        EXPECT_EQ(t1(j, i, k), t1_transposed(i, j, k));
      }
    }
  }

  T t1_unfolded = T::unfold(t1, 1, 3, 2);
  EXPECT_EQ(4, t1_unfolded.dimension());
  EXPECT_EQ(10, t1_unfolded.size(0));
  EXPECT_EQ(9, t1_unfolded.size(1));
  EXPECT_EQ(3, t1_unfolded.size(2));
  EXPECT_EQ(7, t1_unfolded.size(3));
  EXPECT_EQ(t1.storage(), t1_unfolded.storage());
  EXPECT_FALSE(t1_unfolded.isContiguous());
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 9; ++j) {
      for (int k = 0; k < 3; ++k) {
        for (int l = 0; l < 7; ++l) {
          EXPECT_EQ(t1(i, 2 * j + k, l), t1_unfolded(i, j, k, l));
        }
      }
    }
  }

  T t1_cloned = T::clone(t1);
  EXPECT_EQ(t1.dimension(), t1_cloned.dimension());
  for (int i = 0; i < t1.dimension(); ++i) {
    EXPECT_EQ(t1.size(i), t1_cloned.size(i));
  }
  EXPECT_NE(t1.storage(), t1_cloned.storage());
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      for (int k = 0; k < 7; ++k) {
        EXPECT_EQ(t1(i, j, k), t1_cloned(i, j, k));
      }
    }
  }

  T t2(10, 14, 7);
  int t2_val = -200;
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t2_val++);
  }
  T t3 = T::cat(t1, t2, 1);
  EXPECT_EQ(3, t3.dimension());
  EXPECT_EQ(10, t3.size(0));
  EXPECT_EQ(34, t3.size(1));
  EXPECT_EQ(7, t3.size(2));
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      for (int k = 0; k < 7; ++k) {
        EXPECT_EQ(t1(i, j, k), t3(i, j, k));
      }
    }
  }
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 14; ++j) {
      for (int k = 0; k < 7; ++k) {
        EXPECT_EQ(t2(i, j, k), t3(i, j + 20, k));
      }
    }
  }
}

TEST(TensorTest, transformTest) {
  transformTest< DoubleTensor >();
  transformTest< FloatTensor >();
  transformTest< DoubleComplexTensor >();
  transformTest< FloatComplexTensor >();
}

#define TEST_COMPLEX_TRANSFORM(tfunc, sfunc)                            \
  template< typename T >                                                \
  void tfunc ## Test() {                                                \
    T t1(10, 20, 7);                                                    \
    int t1_val = -800;                                                  \
    for (typename T::reference_iterator begin = t1.reference_begin(),   \
             end = t1.reference_end(); begin != end; ++begin) {         \
      *begin = static_cast< typename T::value_type >(t1_val++) /        \
          static_cast< typename T::value_type >(300);                   \
    }                                                                   \
    T t1_result = T::template tfunc< T >(t1);                           \
    for (typename T::reference_iterator begin = t1.reference_begin(),   \
             end = t1.reference_end(); begin != end; ++begin) {         \
      EXPECT_EQ(                                                        \
          static_cast< typename T::value_type >(::std::sfunc(*begin)),  \
          t1_result(begin.position()));                                 \
    }                                                                   \
                                                                        \
    T t2({10, 20, 7}, {161 , 8, 1});                                    \
    int t2_val = -800;                                                  \
    for (typename T::reference_iterator begin = t2.reference_begin(),   \
             end = t2.reference_end(); begin != end; ++begin) {         \
      *begin = static_cast< typename T::value_type >(t2_val++) /        \
          static_cast< typename T::value_type >(300);                   \
    }                                                                   \
    T t2_result = T::template tfunc< T >(t2);                           \
    for (typename T::reference_iterator begin = t2.reference_begin(),   \
             end = t2.reference_end(); begin != end; ++begin) {         \
      EXPECT_EQ(                                                        \
          static_cast< typename T::value_type >(::std::sfunc(*begin)),  \
          t2_result(begin.position()));                                 \
    }                                                                   \
                                                                        \
    T t3(10, 20, 7);                                                    \
    int t3_val = -800;                                                  \
    for (typename T::reference_iterator begin = t3.reference_begin(),   \
             end = t3.reference_end(); begin != end; ++begin) {         \
      *begin = static_cast< typename T::value_type >(t3_val++) /        \
          static_cast< typename T::value_type >(300);                   \
    }                                                                   \
    DoubleTensor t3_result = T::template tfunc< DoubleTensor >(t3);     \
    for (typename T::reference_iterator begin = t3.reference_begin(),   \
             end = t3.reference_end(); begin != end; ++begin) {         \
      EXPECT_EQ(                                                        \
          static_cast< double >(::std::sfunc(*begin)),                  \
          t3_result(begin.position()));                                 \
    }                                                                   \
                                                                        \
    T t4({10, 20, 7}, {161 , 8, 1});                                    \
    int t4_val = -800;                                                  \
    for (typename T::reference_iterator begin = t4.reference_begin(),   \
             end = t4.reference_end(); begin != end; ++begin) {         \
      *begin = static_cast< typename T::value_type >(t4_val++) /        \
          static_cast< typename T::value_type >(300);                   \
    }                                                                   \
    DoubleTensor t4_result = T::template tfunc< DoubleTensor >(t2);     \
    for (typename T::reference_iterator begin = t4.reference_begin(),   \
             end = t4.reference_end(); begin != end; ++begin) {         \
      EXPECT_EQ(                                                        \
          static_cast< double >(::std::sfunc(*begin)),                  \
          t4_result(begin.position()));                                 \
    }                                                                   \
  }                                                                     \
  TEST(TensorTest, tfunc ## Test) {                                     \
    tfunc ## Test< DoubleTensor >();                                    \
    tfunc ## Test< FloatTensor >();                                     \
    tfunc ## Test< DoubleComplexTensor >();                             \
    tfunc ## Test< FloatComplexTensor >();                              \
  }

TEST_COMPLEX_TRANSFORM(getReal, real);
TEST_COMPLEX_TRANSFORM(getImag, imag);
TEST_COMPLEX_TRANSFORM(getArg, arg);
TEST_COMPLEX_TRANSFORM(getCnrm, norm);

#undef TEST_COMPLEX_TRANSFORM

}  // namespace
}  // namespace thunder
