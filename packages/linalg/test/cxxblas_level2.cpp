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
 * variants of order, transpose, uppper-lower and diagonal choices. We always
 * use row-major, no transpose, upper matrix and non-unit diagonal settings.
 */

#include <complex>
#include <random>

#include "gtest/gtest.h"
#include "thunder/linalg/cxxblas.hpp"

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

  D a[m * lda], x[n * incx], y[m * incy], y_orig[m * incy];
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
        result += a[i * lda + j] * x[x_index * incx];
      }
    }
    result = alpha * result + beta * y_orig[i * incy];
    expectEq(result, y[i * incy]);
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
  const int lda = 54;
  const int incx = 2;
  const int incy = 3;

  D a[m * lda], x[n * incx], y[m * incy], y_orig[m * incy];
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
      result += a[i * lda + j] * x[j * incx];
    }
    result = alpha * result + beta * y_orig[i * incy];
    expectEq(result, y[i * incy]);
  }
}

TEST(CxxBlasTest, gemvTest) {
  gemvTest< double >();
  gemvTest< float >();
  gemvTest< ::std::complex< double > >();
  gemvTest< ::std::complex< float > >();
}

template < typename D >
void gerTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int m = 23;
  const int n = 40;
  const int lda = 54;
  const int incx = 2;
  const int incy = 3;

  D a[m * lda], a_orig[lda * m], x[m * incx], y[n * incy];
  D alpha = rand(&gen, &dist);
  D beta = rand(&gen, &dist);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i * lda + j] = rand(&gen, &dist);
      a_orig[i * lda + j] = a[i * lda + j];
    }
  }
  for (int i = 0; i < m; ++i) {
    x[i * incx] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    y[i * incy] = rand(&gen, &dist);
  }
  cxxblas::ger(m, n, x, y, a, alpha, incx, incy, lda);

  D result = 0.0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      result = alpha * x[i * incx] * y[j * incy] + a_orig[i * lda + j];
      expectEq(result, a[i * lda + j]);
    }
  }
}

TEST(CxxBlasTest, gerTest) {
  gerTest< double >();
  gerTest< float >();
  gerTest< ::std::complex< double > >();
  gerTest< ::std::complex< float > >();
}

template < typename D >
void gercTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int m = 23;
  const int n = 40;
  const int lda = 54;
  const int incx = 2;
  const int incy = 3;

  D a[m * lda], a_orig[lda * m], x[m * incx], y[n * incy];
  D alpha = rand(&gen, &dist);
  D beta = rand(&gen, &dist);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i * lda + j] = rand(&gen, &dist);
      a_orig[i * lda + j] = a[i * lda + j];
    }
  }
  for (int i = 0; i < m; ++i) {
    x[i * incx] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    y[i * incy] = rand(&gen, &dist);
  }
  cxxblas::gerc(m, n, x, y, a, alpha, incx, incy, lda);

  D result = 0.0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      result = alpha * x[i * incx] * conjg(y[j * incy]) +
          a_orig[i * lda + j];
      expectEq(result, a[i * lda + j]);
    }
  }
}

TEST(CxxBlasTest, gercTest) {
  gercTest< double >();
  gercTest< float >();
  gercTest< ::std::complex< double > >();
  gercTest< ::std::complex< float > >();
}

template < typename D >
void geruTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int m = 23;
  const int n = 40;
  const int lda = 54;
  const int incx = 2;
  const int incy = 3;

  D a[m *  lda], a_orig[lda * m], x[m * incx], y[n * incy];
  D alpha = rand(&gen, &dist);
  D beta = rand(&gen, &dist);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      a[i * lda + j] = rand(&gen, &dist);
      a_orig[i * lda + j] = a[i * lda + j];
    }
  }
  for (int i = 0; i < m; ++i) {
    x[i * incx] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    y[i * incy] = rand(&gen, &dist);
  }
  cxxblas::geru(m, n, x, y, a, alpha, incx, incy, lda);

  D result = 0.0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      result = alpha * x[i * incx] * y[j * incy] + a_orig[i * lda + j];
      expectEq(result, a[i * lda + j]);
    }
  }
}

TEST(CxxBlasTest, geruTest) {
  geruTest< double >();
  geruTest< float >();
  geruTest< ::std::complex< double > >();
  geruTest< ::std::complex< float > >();
}

template < typename D >
void hbmvTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 40;
  const int k = 3;
  const int lda = 8;
  const int incx = 2;
  const int incy = 3;

  D a[n * lda], x[n * incx], y[n * incy], y_orig[n * incy];
  D alpha = rand(&gen, &dist);
  D beta = rand(&gen, &dist);
  for (int i = 0; i < n * lda; ++i) {
    a[i] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    a[i * lda] = ::std::real(a[i * lda]);
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    y[i * incy] = rand(&gen, &dist);
    y_orig[i * incy] = y[i * incy];
  }
  cxxblas::hbmv(n, a, x, y, alpha, beta, k, lda, incx, incy);

  D result = 0.0;
  int x_index = 0;
  for (int i = 0; i < n; ++i) {
    result = a[i * lda] * x[i * incx];
    for (int j = 1; j < k + 1; ++j) {
      // Upper triangle values: using a[i * lda + j]
      x_index = i + j;
      if (x_index >= 0 && x_index < n) {
        result += a[i * lda + j] * x[x_index * incx];
      }
      // Lower triangle values: using conjugate of a[x_index * lda + j]
      x_index = i - j;
      if (x_index >= 0 && x_index < n) {
        result += conjg(a[x_index * lda + j]) * x[x_index * incx];
      }
    }
    result = alpha * result + beta * y_orig[i * incy];
    expectEq(result, y[i * incy]);
  }
}

TEST(CxxBlasTest, hbmvTest) {
  hbmvTest< double >();
  hbmvTest< float >();
  hbmvTest< ::std::complex< double > >();
  hbmvTest< ::std::complex< float > >();
}

template < typename D >
void hemvTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 19;
  const int lda = 28;
  const int incx = 2;
  const int incy = 3;

  D a[n * lda], x[n * incx], y[n * incy], y_orig[n * incy];
  D alpha = rand(&gen, &dist);
  D beta = rand(&gen, &dist);
  for (int i = 0; i < n * lda; ++i) {
    a[i] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    a[i * lda + i] = ::std::real(a[i * lda]);
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    y[i * incy] = rand(&gen, &dist);
    y_orig[i * incy] = y[i * incy];
  }
  cxxblas::hemv(n, a, x, y, alpha, beta, lda, incx, incy);

  D result = 0.0;
  int x_index = 0;
  for (int i = 0; i < n; ++i) {
    result = a[i * lda + i] * x[i * incx];
    for (int j = i + 1; j < n; ++j) {
      // Upper triangle values: using a[i * lda + j]
      result += a[i * lda + j] * x[j * incx];
    }
    for (int j = 0; j < i; ++j) {
      // Lower triangle values: using a[j * lda + i] conjugate
      result += conjg(a[j * lda + i]) * x[j * incx];
    }
    result = alpha * result + beta * y_orig[i * incy];
    expectEq(result, y[i * incy]);
  }
}

TEST(CxxBlasTest, hemvTest) {
  hemvTest< double >();
  hemvTest< float >();
  hemvTest< ::std::complex< double > >();
  hemvTest< ::std::complex< float > >();
}

template < typename D >
void herTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 19;
  const int lda = 28;
  const int incx = 2;

  D a[n * lda], a_orig[n * lda], x[n * incx];
  D alpha = rand(&gen, &dist);
  for (int i = 0; i < n * lda; ++i) {
    a[i] = rand(&gen, &dist);
    a_orig[i] = a[i];
  }
  for (int i = 0; i < n; ++i) {
    a[i * lda + i] = ::std::real(a[i * lda + i]);
    a_orig[i * lda + i] = a[i * lda + i];
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
  }
  cxxblas::her(n, x, a, ::std::real(alpha), incx, lda);

  D result = 0.0;
  int x_index = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      result = ::std::real(alpha) * x[i * incx] * conjg(x[j * incx]) +
          a_orig[i * lda + j];
      expectEq(result, a[i * lda + j]);
    }
  }
}

TEST(CxxBlasTest, herTest) {
  herTest< double >();
  herTest< float >();
  herTest< ::std::complex< double > >();
  herTest< ::std::complex< float > >();
}

template < typename D >
void her2Test() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 19;
  const int lda = 28;
  const int incx = 2;
  const int incy = 3;

  D a[n * lda], a_orig[n * lda], x[n * incx], y[n * incy];
  D alpha = rand(&gen, &dist);
  for (int i = 0; i < n * lda; ++i) {
    a[i] = rand(&gen, &dist);
    a_orig[i] = a[i];
  }
  for (int i = 0; i < n; ++i) {
    a[i * lda + i] = ::std::real(a[i * lda + i]);
    a_orig[i * lda + i] = a[i * lda + i];
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
    y[i * incy] = rand(&gen, &dist);
  }
  cxxblas::her2(n, x, y, a, alpha, incx, incy, lda);

  D result = 0.0;
  int x_index = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      result = alpha * x[i * incx] * conjg(y[j * incy]) +
          conjg(alpha) * y[i * incy] * conjg(x[j * incx]) +
          a_orig[i * lda + j];
      expectEq(result, a[i * lda + j]);
    }
  }
}

TEST(CxxBlasTest, her2Test) {
  her2Test< double >();
  her2Test< float >();
  her2Test< ::std::complex< double > >();
  her2Test< ::std::complex< float > >();
}

template < typename D >
void hpmvTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 40;
  const int incx = 2;
  const int incy = 3;

  D a[n * (n + 1) / 2], x[n * incx], y[n * incy], y_orig[n * incy];
  D alpha = rand(&gen, &dist);
  D beta = rand(&gen, &dist);
  for (int i = 0; i < n * (n + 1) / 2; ++i) {
    a[i] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    a[i * n - i * (i - 1) / 2] = ::std::real(a[i * n - i * (i - 1) / 2]);
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    y[i * incy] = rand(&gen, &dist);
    y_orig[i * incy] = y[i * incy];
  }
  cxxblas::hpmv(n, a, x, y, alpha, beta, incx, incy);

  D result = 0.0;
  for (int i = 0; i < n; ++i) {
    result = 0.0;
    for (int j = 0; j < i; ++j) {
      // Lower triangle: a(i, j) = conjg(a(j,i))
      result += conjg(a[j * n - j * (j - 1) / 2 + i - j]) * x[j * incx];
    }
    for (int j = i; j < n; ++j) {
      // Upper triangle: using a(i, j)
      result += a[i * n - i * (i - 1) / 2 + j - i] * x[j * incx];
    }
    result = alpha * result + beta * y_orig[i * incy];
    expectEq(result, y[i * incy]);
  }
}

TEST(CxxBlasTest, hpmvTest) {
  hpmvTest< double >();
  hpmvTest< float >();
  hpmvTest< ::std::complex< double > >();
  hpmvTest< ::std::complex< float > >();
}

template < typename D >
void hprTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 40;
  const int incx = 2;

  D a[n * (n + 1) / 2], a_orig[n * (n + 1) / 2], x[n * incx];
  D alpha = rand(&gen, &dist);
  for (int i = 0; i < n * (n + 1) / 2; ++i) {
    a[i] = rand(&gen, &dist);
    a_orig[i] = a[i];
  }
  for (int i = 0; i < n; ++i) {
    a[i * n - i * (i - 1) / 2] = ::std::real(a[i * n - i * (i - 1) / 2]);
    a_orig[i * n - i * (i - 1) / 2] = a[i * n - i * (i - 1) / 2];
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
  }
  cxxblas::hpr(n, x, a, ::std::real(alpha), incx);

  D result = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      result = ::std::real(alpha) * x[i * incx] * conjg(x[j * incx]) +
          a_orig[i * n - i * (i - 1) / 2 + j - i];
      expectEq(result, a[i * n - i * (i - 1) / 2 + j - i]);
    }
  }
}

TEST(CxxBlasTest, hprTest) {
  hprTest< double >();
  hprTest< float >();
  hprTest< ::std::complex< double > >();
  hprTest< ::std::complex< float > >();
}

template < typename D >
void hpr2Test() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 40;
  const int incx = 2;
  const int incy = 3;

  D a[n * (n + 1) / 2], a_orig[n * (n + 1) / 2], x[n * incx], y[n * incy];
  D alpha = rand(&gen, &dist);
  for (int i = 0; i < n * (n + 1) / 2; ++i) {
    a[i] = rand(&gen, &dist);
    a_orig[i] = a[i];
  }
  for (int i = 0; i < n; ++i) {
    a[i * n - i * (i - 1) / 2] = ::std::real(a[i * n - i * (i - 1) / 2]);
    a_orig[i * n - i * (i - 1) / 2] = a[i * n - i * (i - 1) / 2];
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    y[i * incy] = rand(&gen, &dist);
  }
  cxxblas::hpr2(n, x, y, a, alpha, incx, incy);

  D result = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      // Upper triangle: using a(i, j)
      result = alpha * x[i * incx] * conjg(y[j * incy]) +
          conjg(alpha) * y[i * incy] * conjg(x[j * incx]) +
          a_orig[i * n - i * (i - 1) / 2 + j - i];
      expectEq(result, a[i * n - i * (i - 1) / 2 + j - i]);
    }
  }
}

TEST(CxxBlasTest, hpr2Test) {
  hpr2Test< double >();
  hpr2Test< float >();
  hpr2Test< ::std::complex< double > >();
  hpr2Test< ::std::complex< float > >();
}

template < typename D >
void sbmvTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 40;
  const int k = 3;
  const int lda = 8;
  const int incx = 2;
  const int incy = 3;

  D a[n * lda], x[n * incx], y[n * incy], y_orig[n * incy];
  D alpha = rand(&gen, &dist);
  D beta = rand(&gen, &dist);
  for (int i = 0; i < n * lda; ++i) {
    a[i] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    a[i * lda] = ::std::real(a[i * lda]);
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    y[i * incy] = rand(&gen, &dist);
    y_orig[i * incy] = y[i * incy];
  }
  cxxblas::sbmv(n, a, x, y, alpha, beta, k, lda, incx, incy);

  D result = 0.0;
  int x_index = 0;
  for (int i = 0; i < n; ++i) {
    result = a[i * lda] * x[i * incx];
    for (int j = 1; j < k + 1; ++j) {
      // Upper triangle values: using a[i * lda + j]
      x_index = i + j;
      if (x_index >= 0 && x_index < n) {
        result += a[i * lda + j] * x[x_index * incx];
      }
      // Lower triangle values: using conjugate of a[x_index * lda + j]
      x_index = i - j;
      if (x_index >= 0 && x_index < n) {
        result += conjg(a[x_index * lda + j]) * x[x_index * incx];
      }
    }
    result = alpha * result + beta * y_orig[i * incy];
    expectEq(result, y[i * incy]);
  }
}

TEST(CxxBlasTest, sbmvTest) {
  sbmvTest< double >();
  sbmvTest< float >();
  sbmvTest< ::std::complex< double > >();
  sbmvTest< ::std::complex< float > >();
}

template < typename D >
void symvTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 19;
  const int lda = 28;
  const int incx = 2;
  const int incy = 3;

  D a[n * lda], x[n * incx], y[n * incy], y_orig[n * incy];
  D alpha = rand(&gen, &dist);
  D beta = rand(&gen, &dist);
  for (int i = 0; i < n * lda; ++i) {
    a[i] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    a[i * lda + i] = ::std::real(a[i * lda]);
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    y[i * incy] = rand(&gen, &dist);
    y_orig[i * incy] = y[i * incy];
  }
  cxxblas::symv(n, a, x, y, alpha, beta, lda, incx, incy);

  D result = 0.0;
  int x_index = 0;
  for (int i = 0; i < n; ++i) {
    result = a[i * lda + i] * x[i * incx];
    for (int j = i + 1; j < n; ++j) {
      // Upper triangle values: using a[i * lda + j]
      result += a[i * lda + j] * x[j * incx];
    }
    for (int j = 0; j < i; ++j) {
      // Lower triangle values: using a[j * lda + i] conjugate
      result += conjg(a[j * lda + i]) * x[j * incx];
    }
    result = alpha * result + beta * y_orig[i * incy];
    expectEq(result, y[i * incy]);
  }
}

TEST(CxxBlasTest, symvTest) {
  symvTest< double >();
  symvTest< float >();
  symvTest< ::std::complex< double > >();
  symvTest< ::std::complex< float > >();
}

template < typename D >
void syrTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 19;
  const int lda = 28;
  const int incx = 2;

  D a[n * lda], a_orig[n * lda], x[n * incx];
  D alpha = rand(&gen, &dist);
  for (int i = 0; i < n * lda; ++i) {
    a[i] = rand(&gen, &dist);
    a_orig[i] = a[i];
  }
  for (int i = 0; i < n; ++i) {
    a[i * lda + i] = ::std::real(a[i * lda + i]);
    a_orig[i * lda + i] = a[i * lda + i];
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
  }
  cxxblas::syr(n, x, a, ::std::real(alpha), incx, lda);

  D result = 0.0;
  int x_index = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      result = ::std::real(alpha) * x[i * incx] * conjg(x[j * incx]) +
          a_orig[i * lda + j];
      expectEq(result, a[i * lda + j]);
    }
  }
}

TEST(CxxBlasTest, syrTest) {
  syrTest< double >();
  syrTest< float >();
  syrTest< ::std::complex< double > >();
  syrTest< ::std::complex< float > >();
}

template < typename D >
void syr2Test() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 19;
  const int lda = 28;
  const int incx = 2;
  const int incy = 3;

  D a[n * lda], a_orig[n * lda], x[n * incx], y[n * incy];
  D alpha = rand(&gen, &dist);
  for (int i = 0; i < n * lda; ++i) {
    a[i] = rand(&gen, &dist);
    a_orig[i] = a[i];
  }
  for (int i = 0; i < n; ++i) {
    a[i * lda + i] = ::std::real(a[i * lda + i]);
    a_orig[i * lda + i] = a[i * lda + i];
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
    y[i * incy] = rand(&gen, &dist);
  }
  cxxblas::syr2(n, x, y, a, alpha, incx, incy, lda);

  D result = 0.0;
  int x_index = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      result = alpha * x[i * incx] * conjg(y[j * incy]) +
          conjg(alpha) * y[i * incy] * conjg(x[j * incx]) +
          a_orig[i * lda + j];
      expectEq(result, a[i * lda + j]);
    }
  }
}

TEST(CxxBlasTest, syr2Test) {
  syr2Test< double >();
  syr2Test< float >();
  syr2Test< ::std::complex< double > >();
  syr2Test< ::std::complex< float > >();
}

template < typename D >
void spmvTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 40;
  const int incx = 2;
  const int incy = 3;

  D a[n * (n + 1) / 2], x[n * incx], y[n * incy], y_orig[n * incy];
  D alpha = rand(&gen, &dist);
  D beta = rand(&gen, &dist);
  for (int i = 0; i < n * (n + 1) / 2; ++i) {
    a[i] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    a[i * n - i * (i - 1) / 2] = ::std::real(a[i * n - i * (i - 1) / 2]);
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    y[i * incy] = rand(&gen, &dist);
    y_orig[i * incy] = y[i * incy];
  }
  cxxblas::spmv(n, a, x, y, alpha, beta, incx, incy);

  D result = 0.0;
  for (int i = 0; i < n; ++i) {
    result = 0.0;
    for (int j = 0; j < i; ++j) {
      // Lower triangle: a(i, j) = conjg(a(j,i))
      result += conjg(a[j * n - j * (j - 1) / 2 + i - j]) * x[j * incx];
    }
    for (int j = i; j < n; ++j) {
      // Upper triangle: using a(i, j)
      result += a[i * n - i * (i - 1) / 2 + j - i] * x[j * incx];
    }
    result = alpha * result + beta * y_orig[i * incy];
    expectEq(result, y[i * incy]);
  }
}

TEST(CxxBlasTest, spmvTest) {
  spmvTest< double >();
  spmvTest< float >();
  spmvTest< ::std::complex< double > >();
  spmvTest< ::std::complex< float > >();
}

template < typename D >
void sprTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 40;
  const int incx = 2;

  D a[n * (n + 1) / 2], a_orig[n * (n + 1) / 2], x[n * incx];
  D alpha = rand(&gen, &dist);
  for (int i = 0; i < n * (n + 1) / 2; ++i) {
    a[i] = rand(&gen, &dist);
    a_orig[i] = a[i];
  }
  for (int i = 0; i < n; ++i) {
    a[i * n - i * (i - 1) / 2] = ::std::real(a[i * n - i * (i - 1) / 2]);
    a_orig[i * n - i * (i - 1) / 2] = a[i * n - i * (i - 1) / 2];
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
  }
  cxxblas::spr(n, x, a, ::std::real(alpha), incx);

  D result = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      result = ::std::real(alpha) * x[i * incx] * conjg(x[j * incx]) +
          a_orig[i * n - i * (i - 1) / 2 + j - i];
      expectEq(result, a[i * n - i * (i - 1) / 2 + j - i]);
    }
  }
}

TEST(CxxBlasTest, sprTest) {
  sprTest< double >();
  sprTest< float >();
  sprTest< ::std::complex< double > >();
  sprTest< ::std::complex< float > >();
}

template < typename D >
void spr2Test() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 40;
  const int incx = 2;
  const int incy = 3;

  D a[n * (n + 1) / 2], a_orig[n * (n + 1) / 2], x[n * incx], y[n * incy];
  D alpha = rand(&gen, &dist);
  for (int i = 0; i < n * (n + 1) / 2; ++i) {
    a[i] = rand(&gen, &dist);
    a_orig[i] = a[i];
  }
  for (int i = 0; i < n; ++i) {
    a[i * n - i * (i - 1) / 2] = ::std::real(a[i * n - i * (i - 1) / 2]);
    a_orig[i * n - i * (i - 1) / 2] = a[i * n - i * (i - 1) / 2];
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    y[i * incy] = rand(&gen, &dist);
  }
  cxxblas::spr2(n, x, y, a, alpha, incx, incy);

  D result = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      // Upper triangle: using a(i, j)
      result = alpha * x[i * incx] * conjg(y[j * incy]) +
          conjg(alpha) * y[i * incy] * conjg(x[j * incx]) +
          a_orig[i * n - i * (i - 1) / 2 + j - i];
      expectEq(result, a[i * n - i * (i - 1) / 2 + j - i]);
    }
  }
}

TEST(CxxBlasTest, spr2Test) {
  spr2Test< double >();
  spr2Test< float >();
  spr2Test< ::std::complex< double > >();
  spr2Test< ::std::complex< float > >();
}

template < typename D >
void tbmvTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 40;
  const int k = 5;
  const int lda = 8;
  const int incx = 2;

  D a[n * lda], x[n * incx], x_orig[n * incx];
  for (int i = 0; i < n * lda; ++i) {
    a[i] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
    x_orig[i * incx] = x[i * incx];
  }
  cxxblas::tbmv(n, a, x, k, lda, incx);

  D result = 0.0;
  for (int i = 0; i < n; ++i) {
    result = 0.0;
    for (int j = i; j <= i + k && j < n; ++j) {
      result += a[i * lda + j - i] * x_orig[j * incx];
    }
    expectEq(result, x[i * incx]);
  }
}

TEST(CxxBlasTest, tbmvTest) {
  tbmvTest< double >();
  tbmvTest< float >();
  tbmvTest< ::std::complex< double > >();
  tbmvTest< ::std::complex< float > >();
}

template < typename D >
void tbsvTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 40;
  const int k = 5;
  const int lda = 8;
  const int incx = 2;

  D a[n * lda], x[n * incx], x_orig[n * incx];
  for (int i = 0; i < n * lda; ++i) {
    a[i] = rand(&gen, &dist) + static_cast< D >(i);
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
    x_orig[i * incx] = x[i * incx];
  }
  cxxblas::tbsv(n, a, x, k, lda, incx);

  D result = 0.0;
  for (int i = 0; i < n; ++i) {
    result = 0.0;
    for (int j = i; j <= i + k && j < n; ++j) {
      result += a[i * lda + j - i] * x[j * incx];
    }
    expectEq(result, x_orig[i * incx]);
  }
}

TEST(CxxBlasTest, tbsvTest) {
  tbsvTest< double >();
  tbsvTest< float >();
  tbsvTest< ::std::complex< double > >();
  tbsvTest< ::std::complex< float > >();
}

template < typename D >
void tpmvTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 40;
  const int k = 5;
  const int lda = 8;
  const int incx = 2;

  D a[n * (n + 1) / 2], x[n * incx], x_orig[n * incx];
  for (int i = 0; i < n * (n + 1) / 2; ++i) {
    a[i] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
    x_orig[i * incx] = x[i * incx];
  }
  cxxblas::tpmv(n, a, x, incx);

  D result = 0.0;
  for (int i = 0; i < n; ++i) {
    result = 0.0;
    for (int j = i; j < n; ++j) {
      result += a[i * n - i * (i - 1) / 2 + j - i] * x_orig[j * incx];
    }
    expectEq(result, x[i * incx]);
  }
}

TEST(CxxBlasTest, tpmvTest) {
  tpmvTest< double >();
  tpmvTest< float >();
  tpmvTest< ::std::complex< double > >();
  tpmvTest< ::std::complex< float > >();
}

template < typename D >
void tpsvTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 40;
  const int k = 5;
  const int lda = 8;
  const int incx = 2;

  D a[n * (n + 1) / 2], x[n * incx], x_orig[n * incx];
  for (int i = 0; i < n * (n + 1) / 2; ++i) {
    a[i] = rand(&gen, &dist) + static_cast< D >(i);
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
    x_orig[i * incx] = x[i * incx];
  }
  cxxblas::tpsv(n, a, x, incx);

  D result = 0.0;
  for (int i = 0; i < n; ++i) {
    result = 0.0;
    for (int j = i; j < n; ++j) {
      result += a[i * n - i * (i - 1) / 2 + j - i] * x[j * incx];
    }
    expectEq(result, x_orig[i * incx]);
  }
}

TEST(CxxBlasTest, tpsvTest) {
  tpsvTest< double >();
  tpsvTest< float >();
  tpsvTest< ::std::complex< double > >();
  tpsvTest< ::std::complex< float > >();
}

template < typename D >
void trmvTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 40;
  const int k = 5;
  const int lda = 47;
  const int incx = 2;

  D a[n * lda], x[n * incx], x_orig[n * incx];
  for (int i = 0; i < n * lda; ++i) {
    a[i] = rand(&gen, &dist);
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
    x_orig[i * incx] = x[i * incx];
  }
  cxxblas::trmv(n, a, x, lda, incx);

  D result = 0.0;
  for (int i = 0; i < n; ++i) {
    result = 0.0;
    for (int j = i; j < n; ++j) {
      result += a[i * lda + j] * x_orig[j * incx];
    }
    expectEq(result, x[i * incx]);
  }
}

TEST(CxxBlasTest, trmvTest) {
  trmvTest< double >();
  trmvTest< float >();
  trmvTest< ::std::complex< double > >();
  trmvTest< ::std::complex< float > >();
}

template < typename D >
void trsvTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 40;
  const int k = 5;
  const int lda = 47;
  const int incx = 2;

  D a[n * lda], x[n * incx], x_orig[n * incx];
  for (int i = 0; i < n * lda; ++i) {
    a[i] = rand(&gen, &dist) + static_cast< D >(i);
  }
  for (int i = 0; i < n; ++i) {
    x[i * incx] = rand(&gen, &dist);
    x_orig[i * incx] = x[i * incx];
  }
  cxxblas::trsv(n, a, x, lda, incx);

  D result = 0.0;
  for (int i = 0; i < n; ++i) {
    result = 0.0;
    for (int j = i; j < n; ++j) {
      result += a[i * lda + j] * x[j * incx];
    }
    expectEq(result, x_orig[i * incx]);
  }
}

TEST(CxxBlasTest, trsvTest) {
  trsvTest< double >();
  trsvTest< float >();
  trsvTest< ::std::complex< double > >();
  trsvTest< ::std::complex< float > >();
}

}  // namespace
}  // namespace linalg
}  // namespace thunder
