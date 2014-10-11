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
#include <limits>
#include <memory>
#include <typeinfo>

#include "gtest/gtest.h"
#include "thunder/storage.hpp"

namespace thunder {
namespace {

#define TEST_UNARY(func, expr)                                          \
  template< typename T >                                                \
  void func ## Test() {                                                 \
    T t1(10, 20, 7);                                                    \
    int t1_val_real = -800;                                             \
    int t1_val_imag = 977;                                              \
    for (typename T::reference_iterator begin = t1.reference_begin(),   \
             end = t1.reference_end(); begin != end; ++begin) {         \
      *begin = typename T::value_type(t1_val_real++, t1_val_imag--) /   \
          static_cast< typename T::value_type >(300);                   \
    }                                                                   \
    T t1_result = T::func(t1);                                          \
    for (typename T::reference_iterator begin = t1.reference_begin(),   \
              end = t1.reference_end(); begin != end; ++begin) {        \
      if (!::std::isnan(::std::real(t1_result(begin.position()))) &&    \
          !::std::isnan(::std::imag(t1_result(begin.position())))) {    \
        EXPECT_EQ(expr, t1_result(begin.position()));                   \
      }                                                                 \
    }                                                                   \
                                                                        \
    T t2({10, 20, 7}, {161 , 8, 1});                                    \
    int t2_val_real = -800;                                             \
    int t2_val_imag = 787;                                              \
    for (typename T::reference_iterator begin = t2.reference_begin(),   \
             end = t2.reference_end(); begin != end; ++begin) {         \
      *begin = typename T::value_type(t2_val_real++, t2_val_imag--) /   \
          static_cast< typename T::value_type >(300);                   \
    }                                                                   \
    T t2_result = T::func(t2);                                          \
    for (typename T::reference_iterator begin = t2.reference_begin(),   \
              end = t2.reference_end(); begin != end; ++begin) {        \
      if (!::std::isnan(::std::real(t2_result(begin.position()))) &&    \
          !::std::isnan(::std::imag(t2_result(begin.position())))) {    \
        EXPECT_EQ(expr, t2_result(begin.position()));                   \
      }                                                                 \
    }                                                                   \
  }                                                                     \
  TEST(ComplexTest, func ## Test) {                                     \
    func ## Test< DoubleComplexTensor >();                              \
    func ## Test< FloatComplexTensor >();                               \
  }

#define TEST_STD_UNARY(func)                                            \
  TEST_UNARY(func, static_cast< typename T::value_type >(               \
      ::std::func(*begin)));

TEST_STD_UNARY(abs);
TEST_STD_UNARY(exp);
TEST_STD_UNARY(log);
TEST_STD_UNARY(log10);
TEST_STD_UNARY(sqrt);
TEST_STD_UNARY(sin);
TEST_STD_UNARY(cos);
TEST_STD_UNARY(tan);
TEST_STD_UNARY(asin);
TEST_STD_UNARY(acos);
TEST_STD_UNARY(atan);
TEST_STD_UNARY(sinh);
TEST_STD_UNARY(cosh);
TEST_STD_UNARY(tanh);
TEST_STD_UNARY(asinh);
TEST_STD_UNARY(acosh);
TEST_STD_UNARY(atanh);
TEST_STD_UNARY(real);
TEST_STD_UNARY(imag);
TEST_STD_UNARY(arg);

#undef TEST_STD_UNARY

TEST_UNARY(fabs, static_cast< typename T::value_type >(::std::abs(*begin)));
TEST_UNARY(exp2, static_cast< typename T::value_type >(::std::pow(2, *begin)));
TEST_UNARY(expm1, static_cast< typename T::value_type >(
    ::std::exp(*begin) - typename T::value_type(1)));
TEST_UNARY(log1p, static_cast< typename T::value_type >(
    ::std::log(*begin + typename T::value_type(1))));
TEST_UNARY(cbrt, static_cast< typename T::value_type >(
    ::std::pow(*begin, 1/3)));
TEST_UNARY(log2, static_cast< typename T::value_type >(
    ::std::log(*begin)/::std::log(typename T::value_type(2))));
TEST_UNARY(logb, ::std::log(::std::abs(*begin))/::std::log(
    typename T::value_type(
        ::std::numeric_limits<typename T::value_type::value_type >::radix)));

#undef TEST_UNARY

template < typename T >
void errorTest() {
    T t1(10, 20, 7);
    int t1_val = -800;
    for (typename T::reference_iterator begin = t1.reference_begin(),
             end = t1.reference_end(); begin != end; ++begin) {
      *begin = static_cast< typename T::value_type >(t1_val++) /
          static_cast< typename T::value_type >(300);
    }
    T t1_result;
    EXPECT_THROW(t1_result = T::erf(t1), domain_error);
    EXPECT_THROW(t1_result = T::erfc(t1), domain_error);
    EXPECT_THROW(t1_result = T::tgamma(t1), domain_error);
    EXPECT_THROW(t1_result = T::lgamma(t1), domain_error);
    EXPECT_THROW(t1_result = T::ceil(t1), domain_error);
    EXPECT_THROW(t1_result = T::floor(t1), domain_error);
    EXPECT_THROW(t1_result = T::trunc(t1), domain_error);
    EXPECT_THROW(t1_result = T::round(t1), domain_error);
    EXPECT_THROW(t1_result = T::nearbyint(t1), domain_error);
    EXPECT_THROW(t1_result = T::rint(t1), domain_error);
    EXPECT_THROW(t1_result = T::fpclassify(t1), domain_error);
    EXPECT_THROW(t1_result = T::isfinite(t1), domain_error);
    EXPECT_THROW(t1_result = T::isinf(t1), domain_error);
    EXPECT_THROW(t1_result = T::isnan(t1), domain_error);
    EXPECT_THROW(t1_result = T::isnormal(t1), domain_error);
    EXPECT_THROW(t1_result = T::signbit(t1), domain_error);
}

TEST(ComplexTest, errorTest) {
  errorTest< FloatComplexTensor >();
  errorTest< DoubleComplexTensor >();
}

}  // namespace
}  // namespace thunder
