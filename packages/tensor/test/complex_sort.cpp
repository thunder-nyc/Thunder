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

#include <random>

#include "thunder/tensor.hpp"
#include "thunder/exception.hpp"

#include "gtest/gtest.h"

namespace thunder {
namespace {

template< typename T >
void sortTest() {
  T t(10, 9, 20), t_sorted;
  Tensor< typename T::size_storage > t_index;
  EXPECT_THROW(t_sorted = T::sort(t, 1), domain_error);
  EXPECT_THROW(t_sorted = T::sort(t, 1, true), domain_error);
  EXPECT_THROW(t_sorted = T::sort(t, 1, &t_index), domain_error);
  EXPECT_THROW(t_sorted = T::sort(t, 1, &t_index, true), domain_error);
}

TEST(ComplexTest, sortTest) {
  sortTest< DoubleComplexTensor >();
  sortTest< FloatComplexTensor >();
}

}  // namespace
}  // namespace thunder
