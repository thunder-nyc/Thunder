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

#include <complex>
#include <cstdlib>

#include "gtest/gtest.h"
#include "thunder/linalg/cxxblas.hpp"

namespace thunder {
namespace linalg {

TEST(LinalgTest, dummyTest) {
  int n = 1000;
  ::std::complex< double > x[1000];
  ::std::complex< double > y[1000];
  ::std::complex< double > r;

  for (int i = 0; i < n; ++i) {
    x[i] = ::std::complex< double >(0.5, 0);
    y[i] = ::std::complex< double >(2, 0);
  }

  ::std::printf("(%g, %g)\n", r.real(), r.imag());
  r = cxxblas::dotc(n, x, y);
  ::std::printf("(%g, %g)\n", r.real(), r.imag());
  ::std::printf("%g\n", cxxblas::nrm2(n, x));
  ::std::printf("%d\n", cxxblas::iamax(n, x));
  ::std::printf("%g\n", cxxblas::asum(n, x));
}

}  // namespace linalg
}  // namespace thunder
