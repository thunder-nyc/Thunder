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

#define TEST_UNARY(func, expr)                                          \
  template< typename T >                                                \
  void func ## Test() {                                                 \
    T t1(10, 20, 7);                                                    \
    int t1_val = -800;                                                  \
    for (typename T::reference_iterator begin = t1.reference_begin(),   \
             end = t1.reference_end(); begin != end; ++begin) {         \
      *begin = static_cast< typename T::value_type >(t1_val++) /        \
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
    int t2_val = -800;                                                  \
    for (typename T::reference_iterator begin = t2.reference_begin(),   \
             end = t2.reference_end(); begin != end; ++begin) {         \
      *begin = static_cast< typename T::value_type >(t2_val++) /        \
          static_cast< typename T::value_type >(300);                   \
    }                                                                   \
    T t2_result = T::func(t2);                                          \
    for (typename T::reference_iterator begin = t2.reference_begin(),   \
              end = t2.reference_end(); begin != end; ++begin) {        \
      if (!::std::isnan(::std::real(t2_result(begin.position()))) ||     \
          !::std::isnan(::std::imag(t2_result(begin.position())))) {     \
        EXPECT_EQ(expr, t2_result(begin.position()));                   \
      }                                                                 \
    }                                                                   \
  }                                                                     \
  TEST(TensorTest, func ## Test) {                                      \
    func ## Test< DoubleComplexTensor >();                              \
    func ## Test< FloatComplexTensor >();                               \
  }

#define TEST_STD_UNARY(func)                                            \
  TEST_UNARY(func, static_cast< typename T::value_type >(               \
      ::std::func(*begin)));                                            \

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

#undef TEST_UNARY

#define TEST_UNARY_DOMAIN_ERROR(func)                                   \
  template< typename T >                                                \
  void func ## Test() {                                                 \
    T t1(10, 20, 7);                                                    \
    int t1_val = -800;                                                  \
    for (typename T::reference_iterator begin = t1.reference_begin(),   \
             end = t1.reference_end(); begin != end; ++begin) {         \
      *begin = static_cast< typename T::value_type >(t1_val++) /        \
          static_cast< typename T::value_type >(300);                   \
    }                                                                   \
    T t1_result;                                                        \
    EXPECT_THROW(t1_result = T::func(t1), domain_error);                \
                                                                        \
    T t2({10, 20, 7}, {161 , 8, 1});                                    \
    int t2_val = -800;                                                  \
    for (typename T::reference_iterator begin = t2.reference_begin(),   \
             end = t2.reference_end(); begin != end; ++begin) {         \
      *begin = static_cast< typename T::value_type >(t2_val++) /        \
          static_cast< typename T::value_type >(300);                   \
    }                                                                   \
    T t2_result;                                                        \
    EXPECT_THROW(t2_result = T::func(t2), domain_error);                \
  }                                                                     \
  TEST(TensorTest, func ## Test) {                                      \
    func ## Test< DoubleComplexTensor >();                              \
    func ## Test< FloatComplexTensor >();                               \
  }

TEST_UNARY_DOMAIN_ERROR(erf);
TEST_UNARY_DOMAIN_ERROR(erfc);
TEST_UNARY_DOMAIN_ERROR(tgamma);
TEST_UNARY_DOMAIN_ERROR(lgamma);
TEST_UNARY_DOMAIN_ERROR(ceil);
TEST_UNARY_DOMAIN_ERROR(floor);
TEST_UNARY_DOMAIN_ERROR(thunc);
TEST_UNARY_DOMAIN_ERROR(round);
TEST_UNARY_DOMAIN_ERROR(nearbyint);
TEST_UNARY_DOMAIN_ERROR(rint);
TEST_UNARY_DOMAIN_ERROR(fpclassify);
TEST_UNARY_DOMAIN_ERROR(isfinite);
TEST_UNARY_DOMAIN_ERROR(isinf);
TEST_UNARY_DOMAIN_ERROR(isnan);
TEST_UNARY_DOMAIN_ERROR(isnormal);
TEST_UNARY_DOMAIN_ERROR(signbit);

}  // namespace
}  // namespace thunder
