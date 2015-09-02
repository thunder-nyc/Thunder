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

/* Disclaimer: these tests are not complete. In particular, we do not test the
 * variants of order, transpose and uppper-lower choices. We always use
 * row-major, no transpose and upper matrix settings.
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

template < typename D >
void gbmvTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int m = 23;
  const int n = 40;
  const int kl = 3;
  const int ku = 2;
  const int lda = 8;
  const int incx = 2;
  const int incy = 3;

  D a[lda * m], x[n * incx], y[m * incy], y_orig[m * incy];
  D alpha = rand(&gen, &dist);
  D beta = rand(&gen, &dist);
  for (int i = 0; i < lda * m; ++i) {
    a[i] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
  }
  for (int i = 0; i < m; ++i) {
    y[i * incy] = rand(&gen, &dist);
    y_orig[i * incy] = y[i * incy];
  }
  cxxblas::gbmv(m, n, a, x, y, alpha, beta, kl, ku, lda, incx, incy);

  D result = 0.0;
  int x_index = 0;
  for (int i = 0; i < m; ++i) {
    result = 0.0;
    for (int j = 0; j < kl + ku + 1; ++j) {
      x_index = j - kl + i;
      if (x_index >= 0 && x_index < n) {
        result += a[lda * i + j] * x[x_index * incx];
      }
    }
    result = alpha * result + beta * y_orig[i * incy];
    expectEq(y[i * incy], result);
  }
}

TEST(CxxBlasTest, gbmvTest) {
  gbmvTest< double >();
  gbmvTest< float >();
  gbmvTest< ::std::complex< double > >();
  gbmvTest< ::std::complex< float > >();
}

template < typename D >
void gemvTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int m = 23;
  const int n = 40;
  const int kl = 3;
  const int ku = 2;
  const int lda = 54;
  const int incx = 2;
  const int incy = 3;

  D a[lda * m], x[n * incx], y[m * incy], y_orig[m * incy];
  D alpha = rand(&gen, &dist);
  D beta = rand(&gen, &dist);
  for (int i = 0; i < lda * m; ++i) {
    a[i] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
  }
  for (int i = 0; i < m; ++i) {
    y[i * incy] = rand(&gen, &dist);
    y_orig[i * incy] = y[i * incy];
  }
  cxxblas::gemv(m, n, a, x, y, alpha, beta, lda, incx, incy);

  D result = 0.0;
  for (int i = 0; i < m; ++i) {
    result = 0.0;
    for (int j = 0; j < n; ++j) {
      result += a[lda * i + j] * x[j * incx];
    }
    result = alpha * result + beta * y_orig[i * incy];
    expectEq(y[i * incy], result);
  }
}

TEST(CxxBlasTest, gemvTest) {
  gemvTest< double >();
  gemvTest< float >();
  gemvTest< ::std::complex< double > >();
  gemvTest< ::std::complex< float > >();
}

}  // namespace
}  // namespace linalg
}  // namespace thunder
