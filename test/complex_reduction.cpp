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
#include "thunder/exception.hpp"
#include "thunder/storage.hpp"

namespace thunder {
namespace {

#define EXPECT_COMPLEX_EQ(x, y)                         \
  EXPECT_FLOAT_EQ(::std::real(x), ::std::real(y));      \
  EXPECT_FLOAT_EQ(::std::imag(x), ::std::imag(y));

template< typename T >
void reductionTest() {
  T t1(10, 20, 7);
  int t1_val_real = -788;
  int t1_val_imag = 801;
  typename T::value_type t1_sum(0, 0);
  typename T::value_type t1_prod(1, 0);
  typename T::value_type t1_mean(0, 0);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t1_val_real++, t1_val_imag--) /
        static_cast< typename T::value_type >(700);
    t1_sum += *begin;
    t1_prod *= *begin;
  }
  t1_mean = t1_sum / static_cast< typename T::value_type >(t1.length());
  EXPECT_COMPLEX_EQ(t1_sum, T::sum(t1));
  EXPECT_COMPLEX_EQ(t1_prod, T::prod(t1));
  EXPECT_COMPLEX_EQ(t1_mean, T::mean(t1));
  typename T::value_type t1_var(0, 0);
  typename T::value_type t1_std(0, 0);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    t1_var += (*begin - t1_mean) * ::std::conj(*begin - t1_mean);
  }
  t1_var /= static_cast< typename T::value_type >(t1.length());
  t1_std = ::std::sqrt(t1_var);
  EXPECT_COMPLEX_EQ(t1_var, T::var(t1));
  EXPECT_COMPLEX_EQ(t1_std, T::std(t1));

  T t2({10, 20, 7}, {161, 8, 1});
  int t2_val_real = -778;
  int t2_val_imag = 811;
  typename T::value_type t2_sum(0, 0);
  typename T::value_type t2_prod(1, 0);
  typename T::value_type t2_mean(0, 0);
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t2_val_real++, t2_val_imag--) /
        static_cast< typename T::value_type >(700);
    t2_sum += *begin;
    t2_prod *= *begin;
  }
  t2_mean = t2_sum / static_cast< typename T::value_type >(t2.length());
  EXPECT_COMPLEX_EQ(t2_sum, T::sum(t2));
  EXPECT_COMPLEX_EQ(t2_prod, T::prod(t2));
  EXPECT_COMPLEX_EQ(t2_mean, T::mean(t2));
  typename T::value_type t2_var(0, 0);
  typename T::value_type t2_std(0, 0);
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    t2_var += (*begin - t2_mean) * ::std::conj(*begin - t2_mean);
  }
  t2_var /= static_cast< typename T::value_type >(t2.length());
  t2_std = ::std::sqrt(t2_var);
  EXPECT_COMPLEX_EQ(t2_var, T::var(t2));
  EXPECT_COMPLEX_EQ(t2_std, T::std(t2));
}

TEST(ComplexTest, reductionTest) {
  reductionTest< DoubleComplexTensor >();
  reductionTest< FloatComplexTensor >();
}

#define TEST_DIM_REDUCTION(func)                                        \
  template < typename T >                                               \
  void func ## Test() {                                                 \
    T t1(10, 20, 7);                                                    \
    int t1_val_real = -788;                                             \
    int t1_val_imag = 801;                                              \
    for (typename T::reference_iterator begin = t1.reference_begin(),   \
             end = t1.reference_end(); begin != end; ++begin) {         \
      *begin = typename T::value_type(t1_val_real++, t1_val_imag--) /   \
          static_cast< typename T::value_type >(300);                   \
    }                                                                   \
                                                                        \
    T t1_result = T::func(t1, 1);                                       \
    EXPECT_EQ(3, t1_result.dimension());                                \
    EXPECT_EQ(10, t1_result.size(0));                                   \
    EXPECT_EQ(1, t1_result.size(1));                                    \
    EXPECT_EQ(7, t1_result.size(2));                                    \
    for (int i = 0; i < 10; ++i) {                                      \
      for (int j = 0; j < 7; ++j) {                                     \
        EXPECT_EQ(t1[i].select(1, j).func(), t1_result(i, 0, j));       \
      }                                                                 \
    }                                                                   \
                                                                        \
    T t2({10, 20, 7, 9}, {1431, 71, 10, 1});                            \
    int t2_val_real = -754;                                             \
    int t2_val_imag = 903;                                              \
    for (typename T::reference_iterator begin = t2.reference_begin(),   \
             end = t2.reference_end(); begin != end; ++begin) {         \
      *begin = typename T::value_type(t2_val_real++, t2_val_imag--) /   \
          static_cast< typename T::value_type >(300);                   \
    }                                                                   \
                                                                        \
    T t2_result = T::func(t2, 1);                                       \
    EXPECT_EQ(4, t2_result.dimension());                                \
    EXPECT_EQ(10, t2_result.size(0));                                   \
    EXPECT_EQ(1, t2_result.size(1));                                    \
    EXPECT_EQ(7, t2_result.size(2));                                    \
    EXPECT_EQ(9, t2_result.size(3));                                    \
    for (int i = 0; i < 10; ++i) {                                      \
      for (int j = 0; j < 7; ++j) {                                     \
        for (int k = 0; k < 9; ++k) {                                   \
          EXPECT_EQ(t2[i].select(1, j).select(1, k).func(),             \
                    t2_result(i, 0, j, k));                             \
        }                                                               \
      }                                                                 \
    }                                                                   \
  }                                                                     \
  TEST(TensorTest, func ## Test) {                                      \
    func ## Test< DoubleComplexTensor >();                              \
    func ## Test< FloatComplexTensor >();                               \
  }

TEST_DIM_REDUCTION(sum);
TEST_DIM_REDUCTION(prod);
TEST_DIM_REDUCTION(mean);
TEST_DIM_REDUCTION(var);
TEST_DIM_REDUCTION(std);

}  // namespace
}  // namespace thunder
