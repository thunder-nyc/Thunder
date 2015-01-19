/*
 * \copyright Copyright 2015 Xiang Zhang All Rights Reserved.
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
#include <random>

#include "gtest/gtest.h"
#include "thunder/linalg/cxxblas.hpp"

namespace thunder {
namespace linalg {
namespace {

void expectEq(const float &a, const float &b) {
  EXPECT_FLOAT_EQ(a, b);
}
void expectEq(const double &a, const double &b) {
  EXPECT_FLOAT_EQ(a, b);
}
void expectEq(
    const ::std::complex< float > &a, const ::std::complex< float > &b) {
  EXPECT_FLOAT_EQ(::std::real(a), ::std::real(b));
  EXPECT_FLOAT_EQ(::std::imag(a), ::std::imag(b));
}
void expectEq(
    const ::std::complex< double > &a, const ::std::complex< double > &b) {
  EXPECT_FLOAT_EQ(::std::real(a), ::std::real(b));
  EXPECT_FLOAT_EQ(::std::imag(a), ::std::imag(b));
}

TEST(CxxBlasTest, gbmvTest) {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  
  const int m = 23;
  const int n = 40;
  const int kl = 3;
  const int ku = 2;
  const int lda = 5;

  float sa[lda * n], sx[::std::max(m, n)], sy[::std::max(m, n)];
  float salpha = static_cast< float >(dist(gen));
  float sbeta = static_cast< float >(dist(gen));
  for (int i = 0; i < lda * n; ++i) {
    sa[i] = static_cast< float >(dist(gen));
  }
  for (int i = 0; i < ::std::max(m, n); ++i) {
    sx[i] = static_cast< float >(dist(gen));
    sy[i] = static_cast< float >(dist(gen));
  }
  cxxblas::gbmv(m, n, sa, sx, sy);
}

}  // namespace
}  // namespace linalg
}  // namespace thunder
