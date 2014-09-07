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
#include <typeinfo>

#include "gtest/gtest.h"
#include "thunder/storage.hpp"

namespace thunder {
namespace {

#define TEST_STD_UNARY(func)                                            \
  template< typename T >                                                \
  void func ## Test() {                                                 \
    T t1(10, 20, 7);                                                    \
    int t1_val = -800;                                                  \
    for (typename T::reference_iterator begin = t1.reference_begin(),   \
             end = t1.reference_end(); begin != end; ++begin) {         \
      *begin = static_cast< typename T::value_type >(t1_val++) / 300;   \
    }                                                                   \
    T t1_result = T::func(t1);                                          \
    for (typename T::reference_iterator begin = t1.reference_begin(),   \
              end = t1.reference_end(); begin != end; ++begin) {        \
      if (::std::isnan(t1_result(begin.position()))) {                  \
        EXPECT_TRUE(::std::isnan(::std::func(*begin)));                 \
      } else {                                                          \
        EXPECT_FLOAT_EQ(::std::func(*begin),                            \
                        t1_result(begin.position()));                   \
      }                                                                 \
    }                                                                   \
                                                                        \
    T t2({10, 20, 7}, {161 , 8, 1});                                    \
    int t2_val = -800;                                                  \
    for (typename T::reference_iterator begin = t2.reference_begin(),   \
             end = t2.reference_end(); begin != end; ++begin) {         \
      *begin = static_cast< typename T::value_type >(t2_val++) / 300;   \
    }                                                                   \
    T t2_result = T::func(t2);                                          \
    for (typename T::reference_iterator begin = t2.reference_begin(),   \
              end = t2.reference_end(); begin != end; ++begin) {        \
      if (::std::isnan(t2_result(begin.position()))) {                  \
        EXPECT_TRUE(::std::isnan(::std::func(*begin)));                 \
      } else {                                                          \
        EXPECT_FLOAT_EQ(::std::func(*begin),                            \
                        t2_result(begin.position()));                   \
      }                                                                 \
    }                                                                   \
  }                                                                     \
  TEST(TensorTest, func ## Test) {                                      \
    func ## Test< DoubleTensor >();                                     \
    func ## Test< FloatTensor >();                                      \
  }

TEST_STD_UNARY(abs);
TEST_STD_UNARY(exp);
TEST_STD_UNARY(exp2);
TEST_STD_UNARY(expm1);
TEST_STD_UNARY(log);
TEST_STD_UNARY(log10);
TEST_STD_UNARY(log2);
TEST_STD_UNARY(log1p);
TEST_STD_UNARY(sqrt);
TEST_STD_UNARY(cbrt);
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
TEST_STD_UNARY(erf);
TEST_STD_UNARY(erfc);
TEST_STD_UNARY(tgamma);
TEST_STD_UNARY(lgamma);
TEST_STD_UNARY(ceil);
TEST_STD_UNARY(floor);
TEST_STD_UNARY(trunc);
TEST_STD_UNARY(round);
TEST_STD_UNARY(nearbyint);
TEST_STD_UNARY(rint);
TEST_STD_UNARY(logb);
TEST_STD_UNARY(fpclassify);
TEST_STD_UNARY(isfinite);
TEST_STD_UNARY(isinf);
TEST_STD_UNARY(isnan);
TEST_STD_UNARY(isnormal);
TEST_STD_UNARY(signbit);
TEST_STD_UNARY(real);
TEST_STD_UNARY(imag);
TEST_STD_UNARY(arg);
TEST_STD_UNARY(conj);
TEST_STD_UNARY(proj);

#undef TEST_STD_UNARY

template< typename T >
void zeroTest() {
  T t1(10, 20, 7);
  int t1_val = -800;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++) / 300;
  }
  T t1_result = T::zero(t1);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(0, t1_result(begin.position()));
  }

  T t2({10, 20, 7}, {161 , 8, 1});
  int t2_val = -800;
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t2_val++) / 300;
  }
  T t2_result = T::zero(t2);
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(0, t2_result(begin.position()));
  }
}
TEST(TensorTest, zeroTest) {
  zeroTest< DoubleTensor >();
  zeroTest< FloatTensor >();
}

template< typename T >
void cnrmTest() {
  T t1(10, 20, 7);
  int t1_val = -800;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++) / 300;
  }
  T t1_result = T::cnrm(t1);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(::std::norm(*begin), t1_result(begin.position()));
  }

  T t2({10, 20, 7}, {161 , 8, 1});
  int t2_val = -800;
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t2_val++) / 300;
  }
  T t2_result = T::cnrm(t2);
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(::std::norm(*begin), t2_result(begin.position()));
  }
}
TEST(TensorTest, cnrmTest) {
  cnrmTest< DoubleTensor >();
  cnrmTest< FloatTensor >();
}

}  // namespace
}  // namespace thunder
