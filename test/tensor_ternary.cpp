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
namespace {

template < typename T >
void fmaTest() {
  T t1(10, 20, 7);
  int t1_val = -800;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++) / 300;
  }
  T t1_result = T::fma(t1, 9, -7);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(
        static_cast< typename T::value_type >(::std::fma(*begin, 9, -7)),
        t1_result(begin.position()));
  }

  T t2(10, 20, 7);
  int t2_val = -744;
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t2_val++) / 392;
  }

  T t2_result = T::fma(t2, t1, -5);
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(
        static_cast< typename T::value_type >(
            ::std::fma(*begin, t1(begin.position()), -5)),
        t2_result(begin.position()));
  }

  T t3(10, 20, 7);
  int t3_val = -944;
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t3_val++) / 392;
  }

  T t3_result = T::fma(t3, 9, t1);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(
        static_cast< typename T::value_type >(
            ::std::fma(*begin, 9, t1(begin.position()))),
        t3_result(begin.position()));
  }

  T t4(10, 20, 7);
  int t4_val = -974;
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t4_val++) / 392;
  }

  T t4_result = T::fma(t4, t1, t2);
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(
        static_cast< typename T::value_type >(
            ::std::fma(*begin, t1(begin.position()), t2(begin.position()))),
        t4_result(begin.position()));
  }

  T t5({10, 20, 7}, {161, 8, 1});
  int t5_val = -877;
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t5_val++) / 488;
  }
  T t5_result = T::fma(t5, -5, 7);
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(
        static_cast< typename T::value_type >(::std::fma(*begin, -5, 7)),
        t5_result(begin.position()));
  }

  T t6({10, 20, 7}, {161, 8, 1});
  int t6_val = -847;
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t6_val++) / 488;
  }
  T t6_result = T::fma(t6, t1, 7);
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(
        static_cast< typename T::value_type >(
            ::std::fma(*begin, t1(begin.position()), 7)),
        t6_result(begin.position()));
  }

  T t7({10, 20, 7}, {161, 8, 1});
  int t7_val = -874;
  for (typename T::reference_iterator begin = t7.reference_begin(),
           end = t7.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t7_val++) / 488;
  }
  T t7_result = T::fma(t7, 4, t1);
  for (typename T::reference_iterator begin = t7.reference_begin(),
           end = t7.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(
        static_cast< typename T::value_type >(
            ::std::fma(*begin, 4, t1(begin.position()))),
        t7_result(begin.position()));
  }

  T t8({10, 20, 7}, {161, 8, 1});
  int t8_val = -899;
  for (typename T::reference_iterator begin = t8.reference_begin(),
           end = t8.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t8_val++) / 488;
  }
  T t8_result = T::fma(t8, t1, t7);
  for (typename T::reference_iterator begin = t8.reference_begin(),
           end = t8.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(
        static_cast< typename T::value_type >(
            ::std::fma(*begin, t1(begin.position()), t7(begin.position()))),
        t8_result(begin.position()));
  }
}

TEST(TensorTest, fmaTest) {
  fmaTest< DoubleTensor >();
  fmaTest< FloatTensor >();
  fmaTest< Tensor< Storage < int > > >();
}

}  // namespace
}  // namespace thunder
