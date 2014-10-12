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
void valueComparisonTest() {
  T t1(10, 20, 7);

  T t2;
  EXPECT_THROW(t2 = (t1 == static_cast< typename T::value_type >(1.4)),
               domain_error);
  T t2_inverse;
  EXPECT_THROW(t2_inverse = (static_cast< typename T::value_type >(1.4) == t1),
               domain_error);

  T t3;
  EXPECT_THROW(t3 = (t1 != static_cast< typename T::value_type >(2.1)),
               domain_error);
  T t3_inverse;
  EXPECT_THROW(t3_inverse = (static_cast< typename T::value_type >(2.1) != t1),
               domain_error);

  T t4;
  EXPECT_THROW(t4 = (t1 > static_cast< typename T::value_type >(-1.1)),
               domain_error);
  T t4_inverse;
  EXPECT_THROW(t4_inverse = (static_cast< typename T::value_type >(-1.1) < t1),
               domain_error);

  T t5;
  EXPECT_THROW(t5 = (t1 < static_cast< typename T::value_type >(1.9)),
               domain_error);
  T t5_inverse;
  EXPECT_THROW(t5_inverse = (static_cast< typename T::value_type >(1.9) > t1),
               domain_error);

  T t6;
  EXPECT_THROW(t6 = (t1 >= static_cast< typename T::value_type >(1.8)),
               domain_error);
  T t6_inverse;
  EXPECT_THROW(t6_inverse = (static_cast< typename T::value_type >(1.8) <= t1),
               domain_error);

  T t7;
  EXPECT_THROW(t7 = (t1 <= static_cast< typename T::value_type >(1.6)),
               domain_error);
  T t7_inverse;
  EXPECT_THROW(t7_inverse = (static_cast< typename T::value_type >(1.6) >= t1),
               domain_error);
}

TEST(ComplexTest, valueComparisonTest) {
  valueComparisonTest< DoubleComplexTensor >();
  valueComparisonTest< FloatComplexTensor >();
}

template< typename T >
void tensorComparisonTest() {
  T t1(10, 20, 7);
  T t2(10, 20, 7);

  T t3;
  EXPECT_THROW(t3 = (t1 == t2), domain_error);
  T t4;
  EXPECT_THROW(t4 = (t1 != t2), domain_error);
  T t5;
  EXPECT_THROW(t5 = (t1 > t2), domain_error);
  T t6;
  EXPECT_THROW(t6 = (t1 > t2), domain_error);
  T t7;
  EXPECT_THROW(t7 = (t1 < t2), domain_error);
  T t8;
  EXPECT_THROW(t8 = (t1 >= t2), domain_error);
  T t9;
  EXPECT_THROW(t9 = (t1 <= t2), domain_error);
}

TEST(ComplexTest, tensorComparisonTest) {
  tensorComparisonTest< DoubleComplexTensor >();
  tensorComparisonTest< FloatComplexTensor >();
}

}  // namespace
}  // namespace thunder
