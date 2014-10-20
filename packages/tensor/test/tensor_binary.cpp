/*
 * \copyright Copyright 2014 Xiang Zhang All Rights Reserved.
 * \license @{
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
 * @}
 */

#include "thunder/tensor.hpp"

#include <memory>
#include <typeinfo>

#include "gtest/gtest.h"
#include "thunder/storage.hpp"

namespace thunder {
namespace {

#define TEST_STD_BINARY(func)                                           \
  template< typename T >                                                \
  void func ## Test() {                                                 \
    T t1(10, 20, 7);                                                    \
    int t1_val = -800;                                                  \
    for (typename T::reference_iterator begin = t1.reference_begin(),   \
             end = t1.reference_end(); begin != end; ++begin) {         \
      *begin = static_cast< typename T::value_type >(t1_val++) / 300;   \
    }                                                                   \
    T t1_result = T::func(t1, 9);                                       \
    for (typename T::reference_iterator begin = t1.reference_begin(),   \
              end = t1.reference_end(); begin != end; ++begin) {        \
      if (::std::isnan(t1_result(begin.position()))) {                  \
        EXPECT_TRUE(::std::isnan(::std::func(*begin, 9)));              \
      } else {                                                          \
        EXPECT_FLOAT_EQ(                                                \
            static_cast< typename T::value_type >(                      \
                ::std::func(*begin, 9)),                                \
            t1_result(begin.position()));                               \
      }                                                                 \
    }                                                                   \
    T t2(10, 20, 7);                                                    \
    int t2_val = -744;                                                  \
    for (typename T::reference_iterator begin = t2.reference_begin(),   \
             end = t2.reference_end(); begin != end; ++begin) {         \
      *begin = static_cast< typename T::value_type >(t2_val++) / 392;   \
    }                                                                   \
                                                                        \
    T t2_result = T::func(t2, t1);                                      \
    for (typename T::reference_iterator begin = t2.reference_begin(),   \
              end = t2.reference_end(); begin != end; ++begin) {        \
      if (::std::isnan(t2_result(begin.position()))) {                  \
        EXPECT_TRUE(::std::isnan(                                       \
            ::std::func(*begin, t1(begin.position()))));                \
      } else {                                                          \
        EXPECT_FLOAT_EQ(                                                \
            static_cast< typename T::value_type >(                      \
                ::std::func(*begin, t1(begin.position()))),             \
            t2_result(begin.position()));                               \
      }                                                                 \
    }                                                                   \
                                                                        \
    T t3({10, 20, 7}, {161 , 8, 1});                                    \
    int t3_val = -800;                                                  \
    for (typename T::reference_iterator begin = t3.reference_begin(),   \
             end = t3.reference_end(); begin != end; ++begin) {         \
      *begin = static_cast< typename T::value_type >(t3_val++) / 488;   \
    }                                                                   \
    T t3_result = T::func(t3, 7);                                       \
    for (typename T::reference_iterator begin = t3.reference_begin(),   \
              end = t3.reference_end(); begin != end; ++begin) {        \
      if (::std::isnan(t3_result(begin.position()))) {                  \
        EXPECT_TRUE(::std::isnan(::std::func(*begin, 7)));              \
      } else {                                                          \
        EXPECT_FLOAT_EQ(                                                \
            static_cast< typename T::value_type >(                      \
                ::std::func(*begin, 7)),                                \
            t3_result(begin.position()));                               \
      }                                                                 \
    }                                                                   \
                                                                        \
    T t4(10, 20, 7);                                                    \
    int t4_val = -744;                                                  \
    for (typename T::reference_iterator begin = t4.reference_begin(),   \
             end = t4.reference_end(); begin != end; ++begin) {         \
      *begin = static_cast< typename T::value_type >(t4_val++) / 392;   \
    }                                                                   \
    T t4_result = T::func(t4, t3);                                      \
    for (typename T::reference_iterator begin = t4.reference_begin(),   \
              end = t4.reference_end(); begin != end; ++begin) {        \
      if (::std::isnan(t4_result(begin.position()))) {                  \
        EXPECT_TRUE(::std::isnan(                                       \
            ::std::func(*begin, t3(begin.position()))));                \
      } else {                                                          \
        EXPECT_FLOAT_EQ(                                                \
            static_cast< typename T::value_type >(                      \
                ::std::func(*begin, t3(begin.position()))),             \
                        t4_result(begin.position()));                   \
      }                                                                 \
    }                                                                   \
  }                                                                     \
  TEST(TensorTest, func ## Test) {                                      \
    func ## Test< DoubleTensor >();                                     \
    func ## Test< FloatTensor >();                                      \
  }

TEST_STD_BINARY(fmod);
TEST_STD_BINARY(remainder);
TEST_STD_BINARY(fmax);
TEST_STD_BINARY(fmin);
TEST_STD_BINARY(fdim);
TEST_STD_BINARY(pow);
TEST_STD_BINARY(hypot);
TEST_STD_BINARY(atan2);
TEST_STD_BINARY(ldexp);
TEST_STD_BINARY(scalbn);
TEST_STD_BINARY(scalbln);
TEST_STD_BINARY(nextafter);
TEST_STD_BINARY(nexttoward);
TEST_STD_BINARY(copysign);
TEST_STD_BINARY(isgreater);
TEST_STD_BINARY(isgreaterequal);
TEST_STD_BINARY(isless);
TEST_STD_BINARY(islessequal);
TEST_STD_BINARY(islessgreater);
TEST_STD_BINARY(isunordered);

#undef TEST_STD_BINARY

template< typename T >
void addTest() {
  T t1(10, 20, 7);
  int t1_val = -800;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++) / 300;
  }
  T t1_result = T::add(t1, 9);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    if (::std::isnan(t1_result(begin.position()))) {
      EXPECT_TRUE(::std::isnan(*begin + 9));
    } else {
      EXPECT_FLOAT_EQ(static_cast< typename T::value_type >(*begin + 9),
                      t1_result(begin.position()));
    }
  }
  T t2(10, 20, 7);
  int t2_val = -744;
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t2_val++) / 392;
  }

  T t2_result = T::add(t2, t1);
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    if (::std::isnan(t2_result(begin.position()))) {
      EXPECT_TRUE(::std::isnan(*begin + t1(begin.position())));
    } else {
      EXPECT_FLOAT_EQ(
          static_cast< typename T::value_type >(*begin + t1(begin.position())),
          t2_result(begin.position()));
    }
  }

  T t3({10, 20, 7}, {161 , 8, 1});
  int t3_val = -800;
  for (typename T::reference_iterator begin = t3.reference_begin(),
         end = t3.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t3_val++) / 488;
  }
  T t3_result = T::add(t3, 7);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    if (::std::isnan(t3_result(begin.position()))) {
      EXPECT_TRUE(::std::isnan(*begin + 7));
  } else {
      EXPECT_FLOAT_EQ(static_cast< typename T::value_type >(*begin + 7),
                      t3_result(begin.position()));
    }
  }

  T t4(10, 20, 7);
  int t4_val = -744;
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t4_val++) / 392;
  }
  T t4_result = T::add(t4, t3);
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    if (::std::isnan(t4_result(begin.position()))) {
      EXPECT_TRUE(::std::isnan(*begin + t3(begin.position())));
    } else {
      EXPECT_FLOAT_EQ(
          static_cast< typename T::value_type >(*begin + t3(begin.position())),
          t4_result(begin.position()));
    }
  }
}
TEST(TensorTest, addTest) {
  addTest< DoubleTensor >();
  addTest< FloatTensor >();
}

template< typename T >
void subTest() {
  T t1(10, 20, 7);
  int t1_val = -800;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++) / 300;
  }
  T t1_result = T::sub(t1, 9);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    if (::std::isnan(t1_result(begin.position()))) {
      EXPECT_TRUE(::std::isnan(*begin - 9));
    } else {
      EXPECT_FLOAT_EQ(static_cast< typename T::value_type >(*begin - 9),
                      t1_result(begin.position()));
    }
  }
  T t2(10, 20, 7);
  int t2_val = -744;
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t2_val++) / 392;
  }

  T t2_result = T::sub(t2, t1);
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    if (::std::isnan(t2_result(begin.position()))) {
      EXPECT_TRUE(::std::isnan(*begin - t1(begin.position())));
    } else {
      EXPECT_FLOAT_EQ(
          static_cast< typename T::value_type >(*begin - t1(begin.position())),
          t2_result(begin.position()));
    }
  }

  T t3({10, 20, 7}, {161 , 8, 1});
  int t3_val = -800;
  for (typename T::reference_iterator begin = t3.reference_begin(),
         end = t3.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t3_val++) / 488;
  }
  T t3_result = T::sub(t3, 7);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    if (::std::isnan(t3_result(begin.position()))) {
      EXPECT_TRUE(::std::isnan(*begin - 7));
  } else {
      EXPECT_FLOAT_EQ(static_cast< typename T::value_type >(*begin - 7),
                      t3_result(begin.position()));
    }
  }

  T t4(10, 20, 7);
  int t4_val = -744;
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t4_val++) / 392;
  }
  T t4_result = T::sub(t4, t3);
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    if (::std::isnan(t4_result(begin.position()))) {
      EXPECT_TRUE(::std::isnan(*begin - t3(begin.position())));
    } else {
      EXPECT_FLOAT_EQ(
          static_cast< typename T::value_type >(*begin - t3(begin.position())),
          t4_result(begin.position()));
    }
  }
}
TEST(TensorTest, subTest) {
  subTest< DoubleTensor >();
  subTest< FloatTensor >();
}

template< typename T >
void mulTest() {
  T t1(10, 20, 7);
  int t1_val = -800;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++) / 300;
  }
  T t1_result = T::mul(t1, 9);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    if (::std::isnan(t1_result(begin.position()))) {
      EXPECT_TRUE(::std::isnan(*begin * 9));
    } else {
      EXPECT_FLOAT_EQ(static_cast< typename T::value_type >(*begin * 9),
                      t1_result(begin.position()));
    }
  }
  T t2(10, 20, 7);
  int t2_val = -744;
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t2_val++) / 392;
  }

  T t2_result = T::mul(t2, t1);
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    if (::std::isnan(t2_result(begin.position()))) {
      EXPECT_TRUE(::std::isnan(*begin * t1(begin.position())));
    } else {
      EXPECT_FLOAT_EQ(
          static_cast< typename T::value_type >(*begin * t1(begin.position())),
          t2_result(begin.position()));
    }
  }

  T t3({10, 20, 7}, {161 , 8, 1});
  int t3_val = -800;
  for (typename T::reference_iterator begin = t3.reference_begin(),
         end = t3.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t3_val++) / 488;
  }
  T t3_result = T::mul(t3, 7);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    if (::std::isnan(t3_result(begin.position()))) {
      EXPECT_TRUE(::std::isnan(*begin * 7));
  } else {
      EXPECT_FLOAT_EQ(static_cast< typename T::value_type >(*begin * 7),
                      t3_result(begin.position()));
    }
  }

  T t4(10, 20, 7);
  int t4_val = -744;
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t4_val++) / 392;
  }
  T t4_result = T::mul(t4, t3);
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    if (::std::isnan(t4_result(begin.position()))) {
      EXPECT_TRUE(::std::isnan(*begin * t3(begin.position())));
    } else {
      EXPECT_FLOAT_EQ(
          static_cast< typename T::value_type >(*begin * t3(begin.position())),
          t4_result(begin.position()));
    }
  }
}
TEST(TensorTest, mulTest) {
  mulTest< DoubleTensor >();
  mulTest< FloatTensor >();
}

template< typename T >
void divTest() {
  T t1(10, 20, 7);
  int t1_val = -800;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++) / 300;
    if (*begin == 0) {
      *begin = 1;
    }
  }
  T t1_result = T::div(t1, 9);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    if (::std::isnan(t1_result(begin.position()))) {
      EXPECT_TRUE(::std::isnan(*begin / 9));
    } else {
      EXPECT_FLOAT_EQ(static_cast< typename T::value_type >(*begin / 9),
                      t1_result(begin.position()));
    }
  }
  T t2(10, 20, 7);
  int t2_val = -744;
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t2_val++) / 392;
  }

  T t2_result = T::div(t2, t1);
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    if (::std::isnan(t2_result(begin.position()))) {
      EXPECT_TRUE(::std::isnan(*begin / t1(begin.position())));
    } else {
      EXPECT_FLOAT_EQ(
          static_cast< typename T::value_type >(*begin / t1(begin.position())),
          t2_result(begin.position()));
    }
  }

  T t3({10, 20, 7}, {161 , 8, 1});
  int t3_val = -800;
  for (typename T::reference_iterator begin = t3.reference_begin(),
         end = t3.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t3_val++) / 488;
    if (*begin == 0) {
      *begin = 1;
    }
  }
  T t3_result = T::div(t3, 7);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    if (::std::isnan(t3_result(begin.position()))) {
      EXPECT_TRUE(::std::isnan(*begin / 7));
  } else {
      EXPECT_FLOAT_EQ(static_cast< typename T::value_type >(*begin / 7),
                      t3_result(begin.position()));
    }
  }

  T t4(10, 20, 7);
  int t4_val = -744;
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t4_val++) / 392;
  }
  T t4_result = T::div(t4, t3);
  for (typename T::reference_iterator begin = t4.reference_begin(),
           end = t4.reference_end(); begin != end; ++begin) {
    if (::std::isnan(t4_result(begin.position()))) {
      EXPECT_TRUE(::std::isnan(*begin / t3(begin.position())));
    } else {
      EXPECT_FLOAT_EQ(
          static_cast< typename T::value_type >(*begin / t3(begin.position())),
          t4_result(begin.position()));
    }
  }
}
TEST(TensorTest, divTest) {
  divTest< DoubleTensor >();
  divTest< FloatTensor >();
}

template< typename T >
void fillTest() {
  T t1(10, 20, 7);
  int t1_val = -800;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++) / 300;
  }
  T t1_result = T::fill(t1, 9);
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(9, t1_result(begin.position()));
  }

  T t3({10, 20, 7}, {161 , 8, 1});
  int t3_val = -800;
  for (typename T::reference_iterator begin = t3.reference_begin(),
         end = t3.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t3_val++) / 488;
  }
  T t3_result = T::fill(t3, 7);
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(7, t3_result(begin.position()));
  }
}
TEST(TensorTest, fillTest) {
  fillTest< DoubleTensor >();
  fillTest< FloatTensor >();
}

template< typename T >
void copyTest() {
  T t1(10, 20, 7);
  int t1_val = -800;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++) / 300;
  }

  T t2(10, 20, 7);
  t2.copy(t1);
  for (typename T::reference_iterator begin = t2.reference_begin(),
           end = t2.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(t1(begin.position()), *begin);
  }

  FloatTensor t3(10, 20, 7);
  t3.copy(t1);
  for (FloatTensor::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(t1(begin.position()), *begin);
  }
}
TEST(TensorTest, copyTest) {
  copyTest< DoubleTensor >();
  copyTest< FloatTensor >();
}

}  // namespace
}  // namespace thunder
