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

#include <complex>
#include <memory>
#include <typeinfo>

#include "gtest/gtest.h"
#include "thunder/storage.hpp"

namespace thunder {
namespace {

template< typename T >
void typeTest() {
  T t1(10, 20, 7);
  int t1_val = 0;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++);
  }

  FloatTensor t2 = static_cast< FloatTensor >(t1);
  EXPECT_NE(static_cast< void* >(t1.storage().get()),
            static_cast< void* >(t2.storage().get()));
  EXPECT_EQ(3, t2.dimension());
  EXPECT_EQ(10, t2.size(0));
  EXPECT_EQ(20, t2.size(1));
  EXPECT_EQ(7, t2.size(2));
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      for (int k = 0; k < 7; ++k) {
        EXPECT_FLOAT_EQ(static_cast< float >(::std::real(t1(i, j, k))),
                        t2(i, j, k));
      }
    }
  }

  T t3({10, 20, 7}, {9, 8, -1});
  int t3_val = 0;
  for (typename T::reference_iterator begin = t3.reference_begin(),
           end = t3.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t3_val++);
  }

  FloatTensor t4 = static_cast< FloatTensor >(t3);
  EXPECT_NE(static_cast< void* >(t3.storage().get()),
            static_cast< void* >(t4.storage().get()));
  EXPECT_EQ(3, t4.dimension());
  EXPECT_EQ(10, t4.size(0));
  EXPECT_EQ(20, t4.size(1));
  EXPECT_EQ(7, t4.size(2));
  EXPECT_EQ(9, t4.stride(0));
  EXPECT_EQ(8, t4.stride(1));
  EXPECT_EQ(-1, t4.stride(2));
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      for (int k = 0; k < 7; ++k) {
        EXPECT_FLOAT_EQ(static_cast< float >(::std::real(t3(i, j, k))),
                        t4(i, j, k));
      }
    }
  }
}

TEST(TensorTest, typeTest) {
  typeTest< DoubleTensor >();
  typeTest< DoubleComplexTensor >();
  typeTest< FloatComplexTensor >();
}

template< typename T >
void applyTest() {
  T t1(10, 20, 7);
  int t1_val = 0;
  double t1_sum = 0.0;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    t1_sum += static_cast< double >(::std::real(t1_val));
    *begin = static_cast< typename T::value_type >(t1_val++);
  }

  double sum = 0;
  T t2 = T::apply(
      t1, ::std::function< typename T::value_type(typename T::value_type x) >(
          [&sum](typename T::value_type x) {
            sum += static_cast< double >(::std::real(x));
            return x + static_cast< typename T::value_type >(1);}));
  EXPECT_FLOAT_EQ(t1_sum, sum);
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      for (int k = 0; k < 7; ++k) {
        EXPECT_EQ(t1(i, j, k) + static_cast< typename T::value_type >(1),
                  t2(i, j, k));
      }
    }
  }

  sum = 0;
  T t3 = T::apply(t1, ::std::function< typename T::value_type(
      const typename T::value_type&) > (
          [&sum](const typename T::value_type &x) {
            sum += static_cast< double >(::std::real(x));
            return x + static_cast< typename T::value_type >(1);} ));
  EXPECT_FLOAT_EQ(t1_sum, sum);
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      for (int k = 0; k < 7; ++k) {
        EXPECT_EQ(t1(i, j, k) + static_cast< typename T::value_type >(1),
                  t3(i, j, k));
      }
    }
  }

  sum = 0;
  T t4 = T::apply(t1, [&sum](typename T::value_type &x) {
      sum += static_cast< double >(::std::real(x));
      x += static_cast< typename T::value_type >(1);
    });
  EXPECT_FLOAT_EQ(t1_sum, sum);
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      for (int k = 0; k < 7; ++k) {
        EXPECT_EQ(t1(i, j, k) + static_cast< typename T::value_type >(1),
                  t4(i, j, k));
      }
    }
  }

  sum = 0;
  T t5 = T::apply(t1, [&sum](typename T::value_type *x) {
      sum += static_cast< double >(::std::real(*x));
      *x += 1;
    });
  EXPECT_FLOAT_EQ(t1_sum, sum);
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      for (int k = 0; k < 7; ++k) {
        EXPECT_EQ(t1(i, j, k) + static_cast< typename T::value_type >(1),
                  t5(i, j, k));
      }
    }
  }
}

TEST(TensorTest, applyTest) {
  applyTest< DoubleTensor >();
  applyTest< FloatTensor >();
  applyTest< DoubleComplexTensor >();
  applyTest< FloatComplexTensor >();
}

template< typename T >
void noncontiguousApplyTest() {
  T t1({10, 20, 7}, {290, 14, 2});
  int t1_val = 0;
  double t1_sum = 0.0;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    t1_sum += static_cast< double >(::std::real(t1_val));
    *begin = static_cast< typename T::value_type >(t1_val++);
  }

  double sum = 0;
  T t2 = T::apply(
      t1, ::std::function< typename T::value_type(typename T::value_type x) >(
          [&sum](typename T::value_type x) {
            sum += static_cast< double >(::std::real(x));
            return x + static_cast< typename T::value_type >(1);}));
  EXPECT_FLOAT_EQ(t1_sum, sum);
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      for (int k = 0; k < 7; ++k) {
        EXPECT_EQ(t1(i, j, k) + static_cast< typename T::value_type >(1),
                  t2(i, j, k));
      }
    }
  }

  sum = 0;
  T t3 = T::apply(t1, ::std::function< typename T::value_type(
      const typename T::value_type&) > (
          [&sum](const typename T::value_type &x) {
            sum += static_cast< double >(::std::real(x));
            return x + static_cast< typename T::value_type >(1);} ));
  EXPECT_FLOAT_EQ(t1_sum, sum);
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      for (int k = 0; k < 7; ++k) {
        EXPECT_EQ(t1(i, j, k) + static_cast< typename T::value_type >(1),
                  t3(i, j, k));
      }
    }
  }

  sum = 0;
  T t4 = T::apply(t1, [&sum](typename T::value_type &x) {
      sum += static_cast< double >(::std::real(x));
      x += static_cast< typename T::value_type >(1);
    });
  EXPECT_FLOAT_EQ(t1_sum, sum);
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      for (int k = 0; k < 7; ++k) {
        EXPECT_EQ(t1(i, j, k) + static_cast< typename T::value_type >(1),
                  t4(i, j, k));
      }
    }
  }

  sum = 0;
  T t5 = T::apply(t1, [&sum](typename T::value_type *x) {
      sum += static_cast< double >(::std::real(*x));
      *x += static_cast< typename T::value_type >(1);
    });
  EXPECT_FLOAT_EQ(t1_sum, sum);
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 20; ++j) {
      for (int k = 0; k < 7; ++k) {
        EXPECT_EQ(t1(i, j, k) + static_cast< typename T::value_type >(1),
                  t5(i, j, k));
      }
    }
  }
}

TEST(TensorTest, noncontiguousApplyTest) {
  noncontiguousApplyTest< DoubleTensor >();
  noncontiguousApplyTest< FloatTensor >();
  noncontiguousApplyTest< DoubleComplexTensor >();
  noncontiguousApplyTest< FloatComplexTensor >();
}

}  // namespace
}  // namespace thunder
