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

#include <complex>
#include <memory>
#include <typeinfo>

#include "gtest/gtest.h"
#include "thunder/storage.hpp"

namespace thunder {
namespace tensor {
template class Tensor< Storage< ::std::complex< double > > >;
}  // namespace tensor
}  // namespace thunder

namespace thunder {
namespace {

template< typename T >
void onesTest() {
  T t1 = T::ones(9);
  EXPECT_EQ(1, t1.dimension());
  EXPECT_EQ(9, t1.size(0));
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(static_cast< typename T::value_type >(1), *begin);
  }

  T t2 = T::ones(9, 7);
  EXPECT_EQ(2, t2.dimension());
  EXPECT_EQ(9, t2.size(0));
  EXPECT_EQ(7, t2.size(1));
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(static_cast< typename T::value_type >(1), *begin);
  }

  T t3 = T::ones(9, 7, 8);
  EXPECT_EQ(3, t3.dimension());
  EXPECT_EQ(9, t3.size(0));
  EXPECT_EQ(7, t3.size(1));
  EXPECT_EQ(8, t3.size(2));
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(static_cast< typename T::value_type >(1), *begin);
  }

  T t4 = T::ones(9, 7, 8, 4);
  EXPECT_EQ(4, t4.dimension());
  EXPECT_EQ(9, t4.size(0));
  EXPECT_EQ(7, t4.size(1));
  EXPECT_EQ(8, t4.size(2));
  EXPECT_EQ(4, t4.size(3));
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(static_cast< typename T::value_type >(1), *begin);
  }

  T t5 = T::ones({9, 7, 8, 4, 5});
  EXPECT_EQ(5, t5.dimension());
  EXPECT_EQ(9, t5.size(0));
  EXPECT_EQ(7, t5.size(1));
  EXPECT_EQ(8, t5.size(2));
  EXPECT_EQ(4, t5.size(3));
  EXPECT_EQ(5, t5.size(4));
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(static_cast< typename T::value_type >(1), *begin);
  }
}

TEST(TensorTest, onesTest) {
  onesTest< DoubleTensor >();
  onesTest< FloatTensor >();
  onesTest< DoubleComplexTensor >();
  onesTest< FloatComplexTensor >();
  onesTest< Tensor< Storage< int > > >();
}

template< typename T >
void zerosTest() {
  T t1 = T::zeros(9);
  EXPECT_EQ(1, t1.dimension());
  EXPECT_EQ(9, t1.size(0));
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(static_cast< typename T::value_type >(0), *begin);
  }

  T t2 = T::zeros(9, 7);
  EXPECT_EQ(2, t2.dimension());
  EXPECT_EQ(9, t2.size(0));
  EXPECT_EQ(7, t2.size(1));
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(static_cast< typename T::value_type >(0), *begin);
  }

  T t3 = T::zeros(9, 7, 8);
  EXPECT_EQ(3, t3.dimension());
  EXPECT_EQ(9, t3.size(0));
  EXPECT_EQ(7, t3.size(1));
  EXPECT_EQ(8, t3.size(2));
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(static_cast< typename T::value_type >(0), *begin);
  }

  T t4 = T::zeros(9, 7, 8, 4);
  EXPECT_EQ(4, t4.dimension());
  EXPECT_EQ(9, t4.size(0));
  EXPECT_EQ(7, t4.size(1));
  EXPECT_EQ(8, t4.size(2));
  EXPECT_EQ(4, t4.size(3));
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(static_cast< typename T::value_type >(0), *begin);
  }

  T t5 = T::zeros({9, 7, 8, 4, 5});
  EXPECT_EQ(5, t5.dimension());
  EXPECT_EQ(9, t5.size(0));
  EXPECT_EQ(7, t5.size(1));
  EXPECT_EQ(8, t5.size(2));
  EXPECT_EQ(4, t5.size(3));
  EXPECT_EQ(5, t5.size(4));
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    EXPECT_EQ(static_cast< typename T::value_type >(0), *begin);
  }
}

TEST(TensorTest, zerosTest) {
  zerosTest< DoubleTensor >();
  zerosTest< FloatTensor >();
  zerosTest< DoubleComplexTensor >();
  zerosTest< FloatComplexTensor >();
  zerosTest< Tensor< Storage< int > > >();
}

template < typename T >
void polarsTest() {
  T t1(5);
  EXPECT_THROW(t1.polars(T::zeros(5), 9), domain_error);
  EXPECT_THROW(t1.polars(5, T::ones(5)), domain_error);
  EXPECT_THROW(t1.polars(T::zeros(5), T::ones(5)), domain_error);
}

TEST(TensorTest, polarsTest) {
  polarsTest< DoubleTensor >();
  polarsTest< FloatTensor >();
  polarsTest< Tensor< Storage< int > > >();
}

}  // namespace
}  // namespace thunder
