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

template< typename T >
void valueNumericalTest() {
  T t1(10, 20, 7);
  int t1_val = -800;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++) / 300;
  }

  T t2 = t1 + static_cast< typename T::value_type >(14);
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(
        t1(begin.position()) + static_cast< typename T::value_type >(14),
        *begin);
  }

  T t3 = static_cast< typename T::value_type >(7) + t1;
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(
        static_cast< typename T::value_type >(7) + t1(begin.position()),
        *begin);
  }

  T t4 = t1 - static_cast< typename T::value_type >(11);
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(
        t1(begin.position()) - static_cast< typename T::value_type >(11),
        *begin);
  }

  T t5 = t1.clone();
  t5 += static_cast< typename T::value_type >(8);
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(
        t1(begin.position()) + static_cast< typename T::value_type >(8),
        *begin);
  }

  T t6 = t1.clone();
  t6 -= static_cast< typename T::value_type >(4);
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(
        t1(begin.position()) - static_cast< typename T::value_type >(4),
        *begin);
  }
}

TEST(TensorTest, valueNumericalTest) {
  valueNumericalTest< DoubleTensor >();
  valueNumericalTest< FloatTensor >();
  valueNumericalTest< Tensor< Storage< int > > >();
}


template< typename T >
void tensorNumericalTest() {
  T t1(10, 20, 7);
  int t1_val = -800;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++) / 300;
  }

  T t2(10, 20, 7);
  int t2_val = -900;
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t2_val++) / 372;
  }

  T t3 = t1 + t2;
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(t1(begin.position()) + t2(begin.position()), *begin);
  }

  T t4 = t1 - t2;
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(t1(begin.position()) - t2(begin.position()), *begin);
  }

  T t5 = t1.clone();
  t5 += t2;
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(t1(begin.position()) + t2(begin.position()), *begin);
  }

  T t6 = t1.clone();
  t6 -= t2;
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(t1(begin.position()) - t2(begin.position()), *begin);
  }
}

TEST(TensorTest, tensorNumericalTest) {
  tensorNumericalTest< DoubleTensor >();
  tensorNumericalTest< FloatTensor >();
  tensorNumericalTest< Tensor< Storage< int > > >();
}

template< typename T >
void valueComparisonTest() {
  T t1(10, 20, 7);
  int t1_val = -800;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++) / 300;
  }

  T t2 = (t1 == static_cast< typename T::value_type >(1.4));
  T t2_inverse = (static_cast< typename T::value_type >(1.4) == t1);
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(*begin, t2_inverse(begin.position()));
    if (t1(begin.position()) == static_cast< typename T::value_type >(1.4)) {
      EXPECT_TRUE(static_cast< bool >(*begin));
    } else {
      EXPECT_FALSE(static_cast< bool >(*begin));
    }
    EXPECT_FLOAT_EQ(*begin, t2_inverse(begin.position()));
  }

  T t3 = (t1 != static_cast< typename T::value_type >(2.1));
  T t3_inverse = (static_cast< typename T::value_type >(2.1) != t1);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    if (t1(begin.position()) != static_cast< typename T::value_type >(2.1)) {
      EXPECT_TRUE(static_cast< bool >(*begin));
    } else {
      EXPECT_FALSE(static_cast< bool >(*begin));
    }
    EXPECT_FLOAT_EQ(*begin, t3_inverse(begin.position()));
  }

  T t4 = (t1 > static_cast< typename T::value_type >(-1.1));
  T t4_inverse = (static_cast< typename T::value_type >(-1.1) < t1);
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    if (t1(begin.position()) > static_cast< typename T::value_type >(-1.1)) {
      EXPECT_TRUE(static_cast< bool >(*begin));
    } else {
      EXPECT_FALSE(static_cast< bool >(*begin));
    }
    EXPECT_FLOAT_EQ(*begin, t4_inverse(begin.position()));
  }

  T t5 = (t1 < static_cast< typename T::value_type >(1.9));
  T t5_inverse = (static_cast< typename T::value_type >(1.9) > t1);
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    if (t1(begin.position()) < static_cast< typename T::value_type >(1.9)) {
      EXPECT_TRUE(static_cast< bool >(*begin));
    } else {
      EXPECT_FALSE(static_cast< bool >(*begin));
    }
    EXPECT_FLOAT_EQ(*begin, t5_inverse(begin.position()));
  }

  T t6 = (t1 >= static_cast< typename T::value_type >(1.8));
  T t6_inverse = (static_cast< typename T::value_type >(1.8) <= t1);
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    if (t1(begin.position()) >= static_cast< typename T::value_type >(1.8)) {
      EXPECT_TRUE(static_cast< bool >(*begin));
    } else {
      EXPECT_FALSE(static_cast< bool >(*begin));
    }
    EXPECT_FLOAT_EQ(*begin, t6_inverse(begin.position()));
  }

  T t7 = (t1 <= static_cast< typename T::value_type >(1.6));
  T t7_inverse = (static_cast< typename T::value_type >(1.6) >= t1);
  for (typename T::reference_iterator begin = t7.reference_begin(),
           end = t7.reference_end(); begin != end; ++begin) {
    if (t1(begin.position()) <= static_cast< typename T::value_type >(1.6)) {
      EXPECT_TRUE(static_cast< bool >(*begin));
    } else {
      EXPECT_FALSE(static_cast< bool >(*begin));
    }
    EXPECT_FLOAT_EQ(*begin, t7_inverse(begin.position()));
  }
}

TEST(TensorTest, valueComparisonTest) {
  valueComparisonTest< DoubleTensor >();
  valueComparisonTest< FloatTensor >();
  valueComparisonTest< Tensor< Storage < int > > >();
}

template< typename T >
void tensorComparisonTest() {
  T t1(10, 20, 7);
  int t1_val = -800;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++) / 300;
  }

  T t2(10, 20, 7);
  int t2_val = -900;
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t2_val++) / 372;
  }

  T t3 = (t1 == t2);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    if (t1(begin.position()) == t2(begin.position())) {
      EXPECT_TRUE(static_cast< bool >(*begin));
    } else {
      EXPECT_FALSE(static_cast< bool >(*begin));
    }
  }

  T t4 = (t1 != t2);
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    if (t1(begin.position()) != t2(begin.position())) {
      EXPECT_TRUE(static_cast< bool >(*begin));
    } else {
      EXPECT_FALSE(static_cast< bool >(*begin));
    }
  }

  T t5 = (t1 > t2);
  for (typename T::reference_iterator begin = t5.reference_begin(),
           end = t5.reference_end(); begin != end; ++begin) {
    if (t1(begin.position()) > t2(begin.position())) {
      EXPECT_TRUE(static_cast< bool >(*begin));
    } else {
      EXPECT_FALSE(static_cast< bool >(*begin));
    }
  }

  T t6 = (t1 > t2);
  for (typename T::reference_iterator begin = t6.reference_begin(),
           end = t6.reference_end(); begin != end; ++begin) {
    if (t1(begin.position()) > t2(begin.position())) {
      EXPECT_TRUE(static_cast< bool >(*begin));
    } else {
      EXPECT_FALSE(static_cast< bool >(*begin));
    }
  }

  T t7 = (t1 < t2);
  for (typename T::reference_iterator begin = t7.reference_begin(),
           end = t7.reference_end(); begin != end; ++begin) {
    if (t1(begin.position()) < t2(begin.position())) {
      EXPECT_TRUE(static_cast< bool >(*begin));
    } else {
      EXPECT_FALSE(static_cast< bool >(*begin));
    }
  }

  T t8 = (t1 >= t2);
  for (typename T::reference_iterator begin = t8.reference_begin(),
           end = t8.reference_end(); begin != end; ++begin) {
    if (t1(begin.position()) >= t2(begin.position())) {
      EXPECT_TRUE(static_cast< bool >(*begin));
    } else {
      EXPECT_FALSE(static_cast< bool >(*begin));
    }
  }

  T t9 = (t1 <= t2);
  for (typename T::reference_iterator begin = t9.reference_begin(),
           end = t9.reference_end(); begin != end; ++begin) {
    if (t1(begin.position()) <= t2(begin.position())) {
      EXPECT_TRUE(static_cast< bool >(*begin));
    } else {
      EXPECT_FALSE(static_cast< bool >(*begin));
    }
  }
}

TEST(TensorTest, tensorComparisonTest) {
  tensorComparisonTest< DoubleTensor >();
  tensorComparisonTest< FloatTensor >();
  tensorComparisonTest< Tensor< Storage < int > > >();
}

}  // namespace
}  // namespace thunder
