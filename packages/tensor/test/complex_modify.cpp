/*
 * \copyright Copyright 2014 - 2015 Xiang Zhang All Rights Reserved.
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

#include "gtest/gtest.h"

namespace thunder {
namespace {

template < typename T >
void viewImagTest() {
  typedef typename T::real_tensor R;
  T t1(10, 20, 7);
  int t1_val = 0;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++);
  }

  R t1_imag = T::viewImag(t1);
  EXPECT_EQ(t1.data(), reinterpret_cast< typename T::pointer >(
      t1_imag.data() - 1));
  EXPECT_EQ(t1.dimension(), t1_imag.dimension());
  for (int i = 0; i < t1.dimension(); ++i) {
    EXPECT_EQ(t1.size(i), t1_imag.size(i));
  }
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    EXPECT_FLOAT_EQ(::std::imag(*begin), t1_imag(begin.position()));
  }
}

TEST(ComplexTest, viewImagTest) {
  viewImagTest< DoubleComplexTensor >();
  viewImagTest< FloatComplexTensor >();
}

}  // namespace
}  // namespace thunder
