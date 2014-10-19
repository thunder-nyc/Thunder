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
  int t1_val_real = -800;
  int t1_val_imag = 811;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t1_val_real++, t1_val_imag--) /
        static_cast< typename T::value_type >(300);
  }
  T t1_result = T::fma(
      t1, typename T::value_type(9, 6), typename T::value_type(-7, 8));
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    typename T::value_type result =
        (*begin) * typename T::value_type(9, 6) + typename T::value_type(-7, 8);
    EXPECT_FLOAT_EQ(::std::real(result),
                    ::std::real(t1_result(begin.position())));
    EXPECT_FLOAT_EQ(::std::imag(result),
                    ::std::imag(t1_result(begin.position())));
  }

  T t2(10, 20, 7);
  int t2_val_real = -744;
  int t2_val_imag = 698;
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t2_val_real++, t2_val_imag--) /
        static_cast< typename T::value_type >(392);
  }

  T t2_result = T::fma(t2, t1, typename T::value_type(-5, 4));
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    typename T::value_type result =
        (*begin) * t1(begin.position()) + typename T::value_type(-5, 4);
    EXPECT_FLOAT_EQ(::std::real(result),
                    ::std::real(t2_result(begin.position())));
    EXPECT_FLOAT_EQ(::std::imag(result),
                    ::std::imag(t2_result(begin.position())));
  }

  T t3(10, 20, 7);
  int t3_val_real = -944;
  int t3_val_imag = 855;
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t3_val_real++, t3_val_imag--) /
        static_cast< typename T::value_type >(392);
  }

  T t3_result = T::fma(t3, typename T::value_type(9, -7), t1);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    typename T::value_type result =
        (*begin) * typename T::value_type(9, -7) + t1(begin.position());
    EXPECT_FLOAT_EQ(::std::real(result),
                    ::std::real(t3_result(begin.position())));
    EXPECT_FLOAT_EQ(::std::imag(result),
                    ::std::imag(t3_result(begin.position())));
  }

  T t4(10, 20, 7);
  int t4_val_real = -974;
  int t4_val_imag = 875;
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t4_val_real++, t4_val_imag--) /
        static_cast< typename T::value_type >(392);
  }

  T t4_result = T::fma(t4, t1, t2);
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    typename T::value_type result =
        (*begin) * t1(begin.position()) + t2(begin.position());
    EXPECT_FLOAT_EQ(::std::real(result),
                    ::std::real(t4_result(begin.position())));
    EXPECT_FLOAT_EQ(::std::imag(result),
                    ::std::imag(t4_result(begin.position())));
  }

  T t5({10, 20, 7}, {161, 8, 1});
  int t5_val_real = -877;
  int t5_val_imag = 645;
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t5_val_real++, t5_val_imag--) /
        static_cast< typename T::value_type >(488);
  }
  T t5_result = T::fma(
      t5, typename T::value_type(-5, 8), typename T::value_type(7, 10));
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    typename T::value_type result = (*begin) * typename T::value_type(-5, 8) +
        typename T::value_type(7, 10);
    EXPECT_FLOAT_EQ(::std::real(result),
                    ::std::real(t5_result(begin.position())));
    EXPECT_FLOAT_EQ(::std::imag(result),
                    ::std::imag(t5_result(begin.position())));
  }

  T t6({10, 20, 7}, {161, 8, 1});
  int t6_val_real = -847;
  int t6_val_imag = 788;
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t6_val_real++, t6_val_imag--) /
        static_cast< typename T::value_type >(488);
  }
  T t6_result = T::fma(t6, t1, typename T::value_type(7, -8));
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    typename T::value_type result = (*begin) * t1(begin.position()) +
        typename T::value_type(7, -8);
    EXPECT_FLOAT_EQ(::std::real(result),
                    ::std::real(t6_result(begin.position())));
    EXPECT_FLOAT_EQ(::std::imag(result),
                    ::std::imag(t6_result(begin.position())));
  }

  T t7({10, 20, 7}, {161, 8, 1});
  int t7_val_real = -874;
  int t7_val_imag = 799;
  for (typename T::reference_iterator begin = t7.reference_begin(),
           end = t7.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t7_val_real++, t7_val_imag--) /
        static_cast< typename T::value_type >(488);
  }
  T t7_result = T::fma(t7, typename T::value_type(4, 7), t1);
  for (typename T::reference_iterator begin = t7.reference_begin(),
           end = t7.reference_end(); begin != end; ++begin) {
    typename T::value_type result = (*begin) * typename T::value_type(4, 7) +
        t1(begin.position());
    EXPECT_FLOAT_EQ(::std::real(result),
                    ::std::real(t7_result(begin.position())));
    EXPECT_FLOAT_EQ(::std::imag(result),
                    ::std::imag(t7_result(begin.position())));
  }

  T t8({10, 20, 7}, {161, 8, 1});
  int t8_val_real = -899;
  int t8_val_imag = 977;
  for (typename T::reference_iterator begin = t8.reference_begin(),
           end = t8.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t8_val_real++, t8_val_imag--) /
        static_cast< typename T::value_type >(488);
  }
  T t8_result = T::fma(t8, t1, t7);
  for (typename T::reference_iterator begin = t8.reference_begin(),
           end = t8.reference_end(); begin != end; ++begin) {
    typename T::value_type result = (*begin) * t1(begin.position()) +
        t7(begin.position());
    EXPECT_FLOAT_EQ(::std::real(result),
                    ::std::real(t8_result(begin.position())));
    EXPECT_FLOAT_EQ(::std::imag(result),
                    ::std::imag(t8_result(begin.position())));
  }
}

TEST(ComplexTest, fmaTest) {
  fmaTest< DoubleComplexTensor >();
  fmaTest< FloatComplexTensor >();
}

template < typename T >
void polarTest() {
  T t1(10, 20, 7);
  int t1_val_real = -800;
  int t1_val_imag = 811;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t1_val_real++, t1_val_imag--) /
        static_cast< typename T::value_type >(300);
  }
  T t1_result = T::polar(
      t1, typename T::value_type(9, 6), typename T::value_type(-7, 8));
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    typename T::value_type result = typename T::value_type(9, 6) * ::std::exp(
        typename T::value_type(-7, 8) * typename T::value_type(0, 1));
    if (!::std::isnan(::std::real(result)) &&
        !::std::isnan(::std::imag(result))) {
      EXPECT_FLOAT_EQ(::std::real(result),
                      ::std::real(t1_result(begin.position())));
      EXPECT_FLOAT_EQ(::std::imag(result),
                      ::std::imag(t1_result(begin.position())));
    }
  }

  T t2(10, 20, 7);
  int t2_val_real = -744;
  int t2_val_imag = 698;
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t2_val_real++, t2_val_imag--) /
        static_cast< typename T::value_type >(392);
  }
  T t2_result = T::polar(t2, t1, typename T::value_type(-5, 4));
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    typename T::value_type result = t1(begin.position()) * ::std::exp(
        typename T::value_type(-5, 4) * typename T::value_type(0, 1));
    if (!::std::isnan(::std::real(result)) &&
        !::std::isnan(::std::imag(result))) {
      EXPECT_FLOAT_EQ(::std::real(result),
                      ::std::real(t2_result(begin.position())));
      EXPECT_FLOAT_EQ(::std::imag(result),
                      ::std::imag(t2_result(begin.position())));
    }
  }

  T t3(10, 20, 7);
  int t3_val_real = -944;
  int t3_val_imag = 855;
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t3_val_real++, t3_val_imag--) /
        static_cast< typename T::value_type >(392);
  }
  T t3_result = T::polar(t3, typename T::value_type(9, -7), t1);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    typename T::value_type result = typename T::value_type(9, -7) * ::std::exp(
        t1(begin.position()) * typename T::value_type(0, 1));
    if (!::std::isnan(::std::real(result)) &&
        !::std::isnan(::std::imag(result))) {
      EXPECT_FLOAT_EQ(::std::real(result),
                      ::std::real(t3_result(begin.position())));
      EXPECT_FLOAT_EQ(::std::imag(result),
                      ::std::imag(t3_result(begin.position())));
    }
  }

  T t4(10, 20, 7);
  int t4_val_real = -974;
  int t4_val_imag = 875;
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t4_val_real++, t4_val_imag--) /
        static_cast< typename T::value_type >(392);
  }
  T t4_result = T::polar(t4, t1, t2);
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    typename T::value_type result = t1(begin.position()) * ::std::exp(
        t2(begin.position()) * typename T::value_type(0, 1));
    if (!::std::isnan(::std::real(result)) &&
        !::std::isnan(::std::imag(result))) {
      EXPECT_FLOAT_EQ(::std::real(result),
                      ::std::real(t4_result(begin.position())));
      EXPECT_FLOAT_EQ(::std::imag(result),
                      ::std::imag(t4_result(begin.position())));
    }
  }

  T t5({10, 20, 7}, {161, 8, 1});
  int t5_val_real = -877;
  int t5_val_imag = 645;
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t5_val_real++, t5_val_imag--) /
        static_cast< typename T::value_type >(488);
  }
  T t5_result = T::polar(
      t5, typename T::value_type(-5, 8), typename T::value_type(7, 10));
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    typename T::value_type result = typename T::value_type(-5, 8) * ::std::exp(
        typename T::value_type(7, 10) * typename T::value_type(0, 1));
    if (!::std::isnan(::std::real(result)) &&
        !::std::isnan(::std::imag(result))) {
      EXPECT_FLOAT_EQ(::std::real(result),
                      ::std::real(t5_result(begin.position())));
      EXPECT_FLOAT_EQ(::std::imag(result),
                      ::std::imag(t5_result(begin.position())));
    }
  }

  T t6({10, 20, 7}, {161, 8, 1});
  int t6_val_real = -847;
  int t6_val_imag = 788;
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t6_val_real++, t6_val_imag--) /
        static_cast< typename T::value_type >(488);
  }
  T t6_result = T::polar(t6, t1, typename T::value_type(7, -8));
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    typename T::value_type result = t1(begin.position()) * ::std::exp(
        typename T::value_type(7, -8) * typename T::value_type(0, 1));
    if (!::std::isnan(::std::real(result)) &&
        !::std::isnan(::std::imag(result))) {
      EXPECT_FLOAT_EQ(::std::real(result),
                      ::std::real(t6_result(begin.position())));
      EXPECT_FLOAT_EQ(::std::imag(result),
                      ::std::imag(t6_result(begin.position())));
    }
  }

  T t7({10, 20, 7}, {161, 8, 1});
  int t7_val_real = -874;
  int t7_val_imag = 799;
  for (typename T::reference_iterator begin = t7.reference_begin(),
           end = t7.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t7_val_real++, t7_val_imag--) /
        static_cast< typename T::value_type >(488);
  }
  T t7_result = T::polar(t7, typename T::value_type(4, 7), t1);
  for (typename T::reference_iterator begin = t7.reference_begin(),
           end = t7.reference_end(); begin != end; ++begin) {
    typename T::value_type result = typename T::value_type(4, 7) * ::std::exp(
        t1(begin.position()) * typename T::value_type(0, 1));
    if (!::std::isnan(::std::real(result)) &&
        !::std::isnan(::std::imag(result))) {
      EXPECT_FLOAT_EQ(::std::real(result),
                      ::std::real(t7_result(begin.position())));
      EXPECT_FLOAT_EQ(::std::imag(result),
                      ::std::imag(t7_result(begin.position())));
    }
  }

  T t8({10, 20, 7}, {161, 8, 1});
  int t8_val_real = -899;
  int t8_val_imag = 977;
  for (typename T::reference_iterator begin = t8.reference_begin(),
           end = t8.reference_end(); begin != end; ++begin) {
    *begin = typename T::value_type(t8_val_real++, t8_val_imag--) /
        static_cast< typename T::value_type >(488);
  }
  T t8_result = T::polar(t8, t1, t7);
  for (typename T::reference_iterator begin = t8.reference_begin(),
           end = t8.reference_end(); begin != end; ++begin) {
    typename T::value_type result = t1(begin.position()) * ::std::exp(
        t7(begin.position()) * typename T::value_type(0, 1));
    if (!::std::isnan(::std::real(result)) &&
        !::std::isnan(::std::imag(result))) {
      EXPECT_FLOAT_EQ(::std::real(result),
                      ::std::real(t8_result(begin.position())));
      EXPECT_FLOAT_EQ(::std::imag(result),
                      ::std::imag(t8_result(begin.position())));
    }
  }

  FloatTensor t9(10, 20, 7);
  int t9_val = -744;
  for (FloatTensor::reference_iterator begin = t9.reference_begin(),
           end = t9.reference_end(); begin != end; ++begin) {
    *begin = t9_val++ / 392;
  }
  T t9_result = T::polar(t1, t9, 4);
  for (FloatTensor::reference_iterator begin = t9.reference_begin(),
           end = t9.reference_end(); begin != end; ++begin) {
    typename T::value_type result = ::std::polar(
        *begin, static_cast< float >(4));
    if (!::std::isnan(::std::real(result)) &&
        !::std::isnan(::std::imag(result))) {
      EXPECT_FLOAT_EQ(::std::real(result),
                      ::std::real(t9_result(begin.position())));
      EXPECT_FLOAT_EQ(::std::imag(result),
                      ::std::imag(t9_result(begin.position())));
    }
  }

  FloatTensor t10(10, 20, 7);
  int t10_val = -879;
  for (FloatTensor::reference_iterator begin = t10.reference_begin(),
           end = t10.reference_end(); begin != end; ++begin) {
    *begin = t10_val++ / 492;
  }
  T t10_result = T::polar(t1, 3, t10);
  for (FloatTensor::reference_iterator begin = t10.reference_begin(),
           end = t10.reference_end(); begin != end; ++begin) {
    typename T::value_type result = ::std::polar(
        static_cast< float >(3), *begin);
    if (!::std::isnan(::std::real(result)) &&
        !::std::isnan(::std::imag(result))) {
      EXPECT_FLOAT_EQ(::std::real(result),
                      ::std::real(t10_result(begin.position())));
      EXPECT_FLOAT_EQ(::std::imag(result),
                      ::std::imag(t10_result(begin.position())));
    }
  }

  FloatTensor t11(10, 20, 7);
  int t11_val = -784;
  for (FloatTensor::reference_iterator begin = t11.reference_begin(),
           end = t11.reference_end(); begin != end; ++begin) {
    *begin = t11_val++ / 432;
  }
  T t11_result = T::polar(t1, t9, t11);
  for (FloatTensor::reference_iterator begin = t11.reference_begin(),
           end = t11.reference_end(); begin != end; ++begin) {
    typename T::value_type result = ::std::polar(t9(begin.position()), *begin);
    if (!::std::isnan(::std::real(result)) &&
        !::std::isnan(::std::imag(result))) {
      EXPECT_FLOAT_EQ(::std::real(result),
                      ::std::real(t11_result(begin.position())));
      EXPECT_FLOAT_EQ(::std::imag(result),
                      ::std::imag(t11_result(begin.position())));
    }
  }

  FloatTensor t12({10, 20, 7}, {161, 8, 1});
  int t12_val = -744;
  for (FloatTensor::reference_iterator begin = t12.reference_begin(),
           end = t12.reference_end(); begin != end; ++begin) {
    *begin = t12_val++ / 392;
  }
  T t12_result = T::polar(t5, t12, 4);
  for (FloatTensor::reference_iterator begin = t12.reference_begin(),
           end = t12.reference_end(); begin != end; ++begin) {
    typename T::value_type result = ::std::polar(
        *begin, static_cast< float >(4));
    if (!::std::isnan(::std::real(result)) &&
        !::std::isnan(::std::imag(result))) {
      EXPECT_FLOAT_EQ(::std::real(result),
                      ::std::real(t12_result(begin.position())));
      EXPECT_FLOAT_EQ(::std::imag(result),
                      ::std::imag(t12_result(begin.position())));
    }
  }

  FloatTensor t13({10, 20, 7}, {161, 8, 1});
  int t13_val = -879;
  for (FloatTensor::reference_iterator begin = t13.reference_begin(),
           end = t13.reference_end(); begin != end; ++begin) {
    *begin = t13_val++ / 492;
  }
  T t13_result = T::polar(t5, 3, t13);
  for (FloatTensor::reference_iterator begin = t13.reference_begin(),
           end = t13.reference_end(); begin != end; ++begin) {
    typename T::value_type result = ::std::polar(
        static_cast< float >(3), *begin);
    if (!::std::isnan(::std::real(result)) &&
        !::std::isnan(::std::imag(result))) {
      EXPECT_FLOAT_EQ(::std::real(result),
                      ::std::real(t13_result(begin.position())));
      EXPECT_FLOAT_EQ(::std::imag(result),
                      ::std::imag(t13_result(begin.position())));
    }
  }

  FloatTensor t14({10, 20, 7}, {161, 8, 1});
  int t14_val = -784;
  for (FloatTensor::reference_iterator begin = t14.reference_begin(),
           end = t14.reference_end(); begin != end; ++begin) {
    *begin = t14_val++ / 432;
  }
  T t14_result = T::polar(t5, t9, t14);
  for (FloatTensor::reference_iterator begin = t14.reference_begin(),
           end = t14.reference_end(); begin != end; ++begin) {
    typename T::value_type result = ::std::polar(t9(begin.position()), *begin);
    if (!::std::isnan(::std::real(result)) &&
        !::std::isnan(::std::imag(result))) {
      EXPECT_FLOAT_EQ(::std::real(result),
                      ::std::real(t14_result(begin.position())));
      EXPECT_FLOAT_EQ(::std::imag(result),
                      ::std::imag(t14_result(begin.position())));
    }
  }
}

TEST(ComplexTest, polarTest) {
  polarTest< DoubleComplexTensor >();
  polarTest< FloatComplexTensor >();
}

}  // namespace
}  // namespace thunder
