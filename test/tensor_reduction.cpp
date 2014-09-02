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
void reductionTest() {
  T t1(10, 20, 7);
  int t1_val = 0;
  for (typename T::reference_iterator begin = t1.reference_begin(),
           end = t1.reference_end(); begin != end; ++begin) {
    *begin = static_cast< typename T::value_type >(t1_val++);
  }

  EXPECT_EQ(t1_val - 1, static_cast< int >(T::max(t1)));
  EXPECT_EQ(0, static_cast<int>(T::min(t1)));
  EXPECT_EQ(t1_val * (t1_val - 1) / 2, static_cast<int>(T::sum(t1)));
  EXPECT_EQ(0, static_cast<int>(T::prod(t1)));
  EXPECT_EQ((t1_val-1)/2, static_cast<int>(T::mean(t1)));
  EXPECT_EQ(652633, static_cast<int>(T::var(t1)));
  EXPECT_EQ(807, static_cast<int>(T::std(t1)));
}

TEST(TensorTest, reductionTest) {
  reductionTest< DoubleTensor >();
  reductionTest< FloatTensor >();
  reductionTest< Tensor< Storage< int > > >();
}

}  // namespace
}  // namespace thunder
