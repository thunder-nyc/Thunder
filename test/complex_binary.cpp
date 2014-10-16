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
#include "thunder/exception.hpp"
#include "thunder/storage.hpp"

namespace thunder {
namespace {

#define TEST_BINARY(func, expr)                                         \
  template < typename D >                                               \
  D func(D x, D y) {                                                    \
    return expr;                                                        \
  }                                                                     \
  template< typename T >                                                \
  void func ## Test() {                                                 \
    T t1(10, 20, 7);                                                    \
    int t1_val_real = -800;                                             \
    int t1_val_imag = 997;                                              \
    for (typename T::reference_iterator begin = t1.reference_begin(),   \
             end = t1.reference_end(); begin != end; ++begin) {         \
      *begin = typename T::value_type(t1_val_real++, t1_val_imag--) /   \
          typename T::value_type(300);                                  \
    }                                                                   \
    T t1_result = T::func(t1, typename T::value_type(9));               \
    for (typename T::reference_iterator begin = t1.reference_begin(),   \
             end = t1.reference_end(); begin != end; ++begin) {         \
      if (!::std::isnan(::std::real(t1_result(begin.position()))) &&    \
          !::std::isnan(::std::imag(t1_result(begin.position())))) {    \
        typename T::value_type result = func(                           \
            *begin, typename T::value_type(9));                         \
        EXPECT_FLOAT_EQ(::std::real(result),                            \
                        ::std::real(t1_result(begin.position())));      \
        EXPECT_FLOAT_EQ(::std::imag(result),                            \
                        ::std::imag(t1_result(begin.position())));      \
      }                                                                 \
    }                                                                   \
    T t2(10, 20, 7);                                                    \
    int t2_val_real = -744;                                             \
    int t2_val_imag = 769;                                              \
    for (typename T::reference_iterator begin = t2.reference_begin(),   \
             end = t2.reference_end(); begin != end; ++begin) {         \
      *begin = typename T::value_type(t2_val_real++, t2_val_imag--) /   \
          typename T::value_type(392);                                  \
    }                                                                   \
                                                                        \
    T t2_result = T::func(t2, t1);                                      \
    for (typename T::reference_iterator begin = t2.reference_begin(),   \
              end = t2.reference_end(); begin != end; ++begin) {        \
      if (!::std::isnan(::std::real(t2_result(begin.position()))) &&    \
          !::std::isnan(::std::imag(t2_result(begin.position())))) {    \
        typename T::value_type result = func(                           \
            *begin, t1(begin.position()));                              \
        EXPECT_FLOAT_EQ(::std::real(result),                            \
                        ::std::real(t2_result(begin.position())));      \
        EXPECT_FLOAT_EQ(::std::imag(result),                            \
                        ::std::imag(t2_result(begin.position())));      \
      }                                                                 \
    }                                                                   \
                                                                        \
    T t3({10, 20, 7}, {161 , 8, 1});                                    \
    int t3_val_real = -800;                                             \
    int t3_val_imag = 855;                                              \
    for (typename T::reference_iterator begin = t3.reference_begin(),   \
             end = t3.reference_end(); begin != end; ++begin) {         \
      *begin = typename T::value_type(t3_val_real++, t3_val_imag--) /   \
          typename T::value_type(488);                                  \
    }                                                                   \
    T t3_result = T::func(t3, typename T::value_type(7));               \
    for (typename T::reference_iterator begin = t3.reference_begin(),   \
              end = t3.reference_end(); begin != end; ++begin) {        \
      if (!::std::isnan(::std::real(t3_result(begin.position()))) &&    \
          !::std::isnan(::std::imag(t3_result(begin.position())))) {    \
        typename T::value_type result = func(                           \
            *begin, typename T::value_type(7));                         \
        EXPECT_FLOAT_EQ(::std::real(result),                            \
                        ::std::real(t3_result(begin.position())));      \
        EXPECT_FLOAT_EQ(::std::imag(result),                            \
                        ::std::imag(t3_result(begin.position())));      \
      }                                                                 \
    }                                                                   \
                                                                        \
    T t4(10, 20, 7);                                                    \
    int t4_val_real = -744;                                             \
    int t4_val_imag = 795;                                              \
    for (typename T::reference_iterator begin = t4.reference_begin(),   \
             end = t4.reference_end(); begin != end; ++begin) {         \
      *begin = typename T::value_type(t4_val_real++, t4_val_imag--) /   \
          typename T::value_type(392);                                  \
    }                                                                   \
    T t4_result = T::func(t4, t3);                                      \
    for (typename T::reference_iterator begin = t4.reference_begin(),   \
              end = t4.reference_end(); begin != end; ++begin) {        \
      if (!::std::isnan(::std::real(t4_result(begin.position()))) &&    \
          !::std::isnan(::std::imag(t4_result(begin.position())))) {    \
        typename T::value_type result = func(                           \
            *begin, t3(begin.position()));                              \
        EXPECT_FLOAT_EQ(::std::real(result),                            \
                        ::std::real(t4_result(begin.position())));      \
        EXPECT_FLOAT_EQ(::std::imag(result),                            \
                        ::std::imag(t4_result(begin.position())));      \
      }                                                                 \
    }                                                                   \
  }                                                                     \
  TEST(ComplexTest, func ## Test) {                                     \
    func ## Test< DoubleComplexTensor >();                              \
    func ## Test< FloatComplexTensor >();                               \
  }

TEST_BINARY(add, x + y);
TEST_BINARY(sub, x - y);
TEST_BINARY(mul, x * y);
TEST_BINARY(div, x / y);
TEST_BINARY(pow, ::std::pow(x, y));
TEST_BINARY(hypot, static_cast< D >(::std::hypot(
    ::std::norm(x), ::std::norm(y))));
TEST_BINARY(atan2, ::std::atan(x / y));
TEST_BINARY(ldexp, x * ::std::pow(static_cast< D >(2), y))
TEST_BINARY(scalbn, x * ::std::pow(static_cast< D >(
    ::std::numeric_limits< typename D::value_type >::radix), y));
TEST_BINARY(scalbln,  x * ::std::pow(static_cast< D >(
    ::std::numeric_limits< typename D::value_type >::radix), y));

#undef TEST_BINARY

template< typename T >
void fillTest() {
  T t1(10, 20, 7);
  int t1_val_real = -800;
  int t1_val_imag = 799;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t1_val_real++, t1_val_imag--) /
        static_cast< typename T::value_type >(300);
  }
  T t1_result = T::fill(t1, typename T::value_type(9, 4));
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(9, ::std::real(t1_result(begin.position())));
    EXPECT_FLOAT_EQ(4, ::std::imag(t1_result(begin.position())));
  }

  T t3({10, 20, 7}, {161 , 8, 1});
  int t3_val_real = -800;
  int t3_val_imag = 877;
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t3_val_real++, t3_val_imag--) /
        static_cast< typename T::value_type >(488);
  }
  T t3_result = T::fill(t3, typename T::value_type(7, 8));
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(7, ::std::real(t3_result(begin.position())));
    EXPECT_FLOAT_EQ(8, ::std::imag(t3_result(begin.position())));
  }
}
TEST(ComplexTest, fillTest) {
  fillTest< DoubleComplexTensor >();
  fillTest< FloatComplexTensor >();
}

template< typename T >
void copyTest() {
  T t1(10, 20, 7);
  int t1_val_real = -800;
  int t1_val_imag = 789;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t1_val_real++, t1_val_imag--) /
        static_cast< typename T::value_type >(300);
  }

  T t2(10, 20, 7);
  t2.copy(t1);
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(::std::real(t1(begin.position())), ::std::real(*begin));
    EXPECT_FLOAT_EQ(::std::imag(t1(begin.position())), ::std::imag(*begin));
  }

  FloatTensor t3(10, 20, 7);
  t3.copy(t1);
  for (FloatTensor::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(::std::real(t1(begin.position())), *begin);
  }

  FloatComplexTensor t4(10, 20, 7);
  t4.copy(t1);
  for (FloatComplexTensor::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(static_cast< float >(::std::real(t1(begin.position()))),
                    ::std::real(*begin));
    EXPECT_FLOAT_EQ(static_cast< float >(::std::imag(t1(begin.position()))),
                    ::std::imag(*begin));
  }
}
TEST(ComplexTest, copyTest) {
  copyTest< DoubleComplexTensor >();
  copyTest< FloatComplexTensor >();
}

template < typename T >
void errorTest() {
    T t1(10, 20, 7);
    int t1_val_real = -800;
    int t1_val_imag = 997;
    for (typename T::reference_iterator begin = t1.reference_begin(),
             end = t1.reference_end(); begin != end; ++begin) {
      *begin = typename T::value_type(t1_val_real++, t1_val_imag--) /
          typename T::value_type(300);
    }
    T t1_result;

    T t2(10, 20, 7);
    int t2_val_real = -744;
    int t2_val_imag = 769;
    for (typename T::reference_iterator begin = t2.reference_begin(),
             end = t2.reference_end(); begin != end; ++begin) {
      *begin = typename T::value_type(t2_val_real++, t2_val_imag--) /
          typename T::value_type(392);
    }
    T t2_result;

    EXPECT_THROW(t1_result = T::fmod(t1, typename T::value_type(9, 4)),
                 domain_error);
    EXPECT_THROW(t2_result = T::fmod(t2, t1), domain_error);

    EXPECT_THROW(t1_result = T::remainder(t1, typename T::value_type(9, 4)),
                 domain_error);
    EXPECT_THROW(t2_result = T::remainder(t2, t1), domain_error);

    EXPECT_THROW(t1_result = T::fmax(t1, typename T::value_type(9, 4)),
                 domain_error);
    EXPECT_THROW(t2_result = T::fmax(t2, t1), domain_error);

    EXPECT_THROW(t1_result = T::fmin(t1, typename T::value_type(9, 4)),
                 domain_error);
    EXPECT_THROW(t2_result = T::fmin(t2, t1), domain_error);

    EXPECT_THROW(t1_result = T::nextafter(t1, typename T::value_type(9, 4)),
                 domain_error);
    EXPECT_THROW(t2_result = T::nextafter(t2, t1), domain_error);

    EXPECT_THROW(t1_result = T::nexttoward(t1, typename T::value_type(9, 4)),
                 domain_error);
    EXPECT_THROW(t2_result = T::nexttoward(t2, t1), domain_error);

    EXPECT_THROW(t1_result = T::copysign(t1, typename T::value_type(9, 4)),
                 domain_error);
    EXPECT_THROW(t2_result = T::copysign(t2, t1), domain_error);

    EXPECT_THROW(t1_result = T::isgreater(t1, typename T::value_type(9, 4)),
                 domain_error);
    EXPECT_THROW(t2_result = T::isgreater(t2, t1), domain_error);

    EXPECT_THROW(t1_result = T::isgreaterequal(
        t1, typename T::value_type(9, 4)), domain_error);
    EXPECT_THROW(t2_result = T::isgreaterequal(t2, t1), domain_error);

    EXPECT_THROW(t1_result = T::isless(t1, typename T::value_type(9, 4)),
                 domain_error);
    EXPECT_THROW(t2_result = T::isless(t2, t1), domain_error);

    EXPECT_THROW(t1_result = T::islessequal(t1, typename T::value_type(9, 4)),
                 domain_error);
    EXPECT_THROW(t2_result = T::islessequal(t2, t1), domain_error);

    EXPECT_THROW(t1_result = T::islessgreater(t1, typename T::value_type(9, 4)),
                 domain_error);
    EXPECT_THROW(t2_result = T::islessgreater(t2, t1), domain_error);

    EXPECT_THROW(t1_result = T::isunordered(t1, typename T::value_type(9, 4)),
                 domain_error);
    EXPECT_THROW(t2_result = T::isunordered(t2, t1), domain_error);
}
TEST(ComplexTest, errorTest) {
  errorTest< DoubleComplexTensor >();
  errorTest< FloatComplexTensor >();
}

}  // namespace
}  // namespace thunder
