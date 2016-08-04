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

#include <complex>
#include <memory>
#include <random>

#include "gtest/gtest.h"
#include "thunder/linalg/cxxblas.hpp"
#include "thunder/tensor.hpp"

namespace thunder {
namespace linalg {
namespace {

template < typename D >
void expectEq(const D &a, const D &b) {
  EXPECT_FLOAT_EQ(a, b);
}
template < typename D >
void expectEq(const ::std::complex< D > &a, const ::std::complex< D > &b) {
  EXPECT_FLOAT_EQ(::std::real(a), ::std::real(b));
  EXPECT_FLOAT_EQ(::std::imag(a), ::std::imag(b));
}

template < typename D >
D conjg(const D &v) {
  return v;
}
template < typename D >
::std::complex< D > conjg(const ::std::complex< D > &v) {
  return ::std::conj(v);
}

template < typename D >
class RandomGenerator {
 public:
  template < typename G, typename R >
  D operator()(G *gen, R *dist) {
    return static_cast< D >((*dist)(*gen));
  }
};

template < typename D >
class RandomGenerator< ::std::complex< D > > {
 public:
  template < typename G, typename R >
  ::std::complex< D > operator()(G *gen, R *dist) {
    return ::std::complex< D >(
        static_cast< D >((*dist)(*gen)), static_cast< D >((*dist)(*gen)));
  }
};

TEST(LinalgTest, gemvTest) {
  EXPECT_EQ(0, 0);
}

}  // namespace
}  // namespace linalg
}  // namespace thunder
