/*
 * \copyright Copyright 2014-2016 Xiang Zhang All Rights Reserved.
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

#include "thunder/linalg.hpp"

#include "gtest/gtest.h"
#include "thunder/exception.hpp"
#include "thunder/linalg/cxxblas.hpp"
#include "thunder/random.hpp"
#include "thunder/tensor.hpp"

namespace thunder {
namespace linalg {
namespace {

template < typename T >
const T& uniformDist(const T& t) {
  Random< typename T::real_tensor > r;
  r.uniform(t.viewReal());
  try {
    r.uniform(t.viewImag());
  } catch (const domain_error &e) {
    // Do nothing for a real tensor type
  }
  return t;
}
template < typename T >
T& uniformDist(T& t) {
  return const_cast< T& >(uniformDist(const_cast< const T& >(t)));
}

TEST(LinalgTest, gemmTest) {
  EXPECT_EQ(0, 0);
}

}  // namespace
}  // namespace linalg
}  // namespace thunder
