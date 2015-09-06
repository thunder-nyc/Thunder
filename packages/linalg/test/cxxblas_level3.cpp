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
 * variants of order, transpose, uppper-lower, diagonal and side choices. We
 * always use row-major, no transpose, upper matrix, non-unit diagonal and left
 * side settings.
 */

#include <complex>
#include <cstdlib>
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
void gemmTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int m = 23;
  const int n = 40;
  const int k = 32;
  const int lda = 39;
  const int ldb = 47;
  const int ldc = 50;
  D a[m * lda], b[k * ldb], c[m * ldc], c_orig[m * ldc];
  D alpha = rand(&gen, &dist);
  D beta = rand(&gen, &dist);
  for (int i = 0; i < m * lda; ++i) {
    a[i] = rand(&gen, &dist);
  }
  for (int i = 0; i < k * ldb; ++i) {
    b[i] = rand(&gen, &dist);
  }
  for (int i = 0; i < m * ldc; ++i) {
    c[i] = rand(&gen, &dist);
    c_orig[i] = c[i];
  }
  cxxblas::gemm(m, n, k, a, b, c, alpha, beta, lda, ldb, ldc);

  D result = 0.0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      result = 0.0;
      for (int p = 0; p < k; ++p) {
        result += a[i * lda + p] * b[p * ldb + j];
      }
      result = alpha * result + beta * c_orig[i * ldc + j];
      expectEq(result, c[i * ldc + j]);
    }
  }
}

TEST(CxxBlasTest, gemmTest) {
  gemmTest< double >();
  gemmTest< float >();
  gemmTest< ::std::complex< double > >();
  gemmTest< ::std::complex< float > >();
}

template < typename D >
void hemmTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int m = 23;
  const int n = 40;
  const int lda = 36;
  const int ldb = 47;
  const int ldc = 50;
  D a[m * lda], b[m * ldb], c[m * ldc], c_orig[m * ldc];
  D alpha = rand(&gen, &dist);
  D beta = rand(&gen, &dist);
  for (int i = 0; i < m * lda; ++i) {
    a[i] = rand(&gen, &dist);
  }
  for (int i = 0; i < m; ++i) {
    a[i * lda + i] = ::std::real(a[i * lda + i]);
  }
  for (int i = 0; i < m * ldb; ++i) {
    b[i] = rand(&gen, &dist);
  }
  for (int i = 0; i < m * ldc; ++i) {
    c[i] = rand(&gen, &dist);
    c_orig[i] = c[i];
  }
  cxxblas::hemm(m, n, a, b, c, alpha, beta, lda, ldb, ldc);

  D result = 0.0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      result = 0.0;
      for (int p = 0; p < i; ++p) {
        // Lower: using a(i, p) = conj(a(p, i))
        result += conjg(a[p * lda + i]) * b[p * ldb + j];
      }
      for (int p = i; p < m; ++p) {
        // Upper: using a(i, p)
        result += a[i * lda + p] * b[p * ldb + j];
      }
      result = alpha * result + beta * c_orig[i * ldc + j];
      expectEq(result, c[i * ldc + j]);
    }
  }
}

TEST(CxxBlasTest, hemmTest) {
  hemmTest< double >();
  hemmTest< float >();
  hemmTest< ::std::complex< double > >();
  hemmTest< ::std::complex< float > >();
}

template < typename D >
void herkTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 23;
  const int k = 15;
  const int lda = 18;
  const int ldc = 33;
  D a[n * lda], c[n * ldc], c_orig[n * ldc];
  D alpha = rand(&gen, &dist);
  D beta = rand(&gen, &dist);
  for (int i = 0; i < n * lda; ++i) {
    a[i] = rand(&gen, &dist);
  }
  for (int i = 0; i < n * ldc; ++i) {
    c[i] = rand(&gen, &dist);
    c_orig[i] = c[i];
  }
  for (int i = 0; i < n; ++i) {
    c[i * ldc + i] = ::std::real(c[i * ldc + i]);
    c_orig[i * ldc + i] = c[i * ldc + i];
  }
  cxxblas::herk(n, k, a, c, ::std::real(alpha), ::std::real(beta), lda, ldc);

  D result = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      result = 0.0;
      for (int p = 0; p < k; ++p) {
        result += a[i * lda + p] * conjg(a[j * lda + p]);
      }
      result = ::std::real(alpha) * result +
          ::std::real(beta) * c_orig[i * ldc + j];
      expectEq(result, c[i * ldc + j]);
    }
  }
}

TEST(CxxBlasTest, herkTest) {
  herkTest< double >();
  herkTest< float >();
  herkTest< ::std::complex< double > >();
  herkTest< ::std::complex< float > >();
}

template < typename D >
void her2kTest() {
  ::std::random_device rd;
  ::std::mt19937 gen(rd());
  ::std::uniform_real_distribution< double > dist(-1.0, 1.0);
  RandomGenerator< D > rand;

  const int n = 23;
  const int k = 15;
  const int lda = 18;
  const int ldb = 25;
  const int ldc = 33;
  D a[n * lda], b[n * ldb], c[n * ldc], c_orig[n * ldc];
  D alpha = rand(&gen, &dist);
  D beta = rand(&gen, &dist);
  for (int i = 0; i < n * lda; ++i) {
    a[i] = rand(&gen, &dist);
  }
  for (int i = 0; i < n * ldb; ++i) {
    b[i] = rand(&gen, &dist);
  }
  for (int i = 0; i < n * ldc; ++i) {
    c[i] = rand(&gen, &dist);
    c_orig[i] = c[i];
  }
  for (int i = 0; i < n; ++i) {
    c[i * ldc + i] = ::std::real(c[i * ldc + i]);
    c_orig[i * ldc + i] = c[i * ldc + i];
  }
  cxxblas::her2k(n, k, a, b, c, alpha, ::std::real(beta), lda, ldb, ldc);

  D result = 0.0;
  for (int i = 0; i < n; ++i) {
    for (int j = i; j < n; ++j) {
      result = 0.0;
      for (int p = 0; p < k; ++p) {
        result += alpha * a[i * lda + p] * conjg(b[j * ldb + p]) +
            conjg(alpha) * b[i * ldb + p] * conjg(a[j * lda + p]);
      }
      result = result + ::std::real(beta) * c_orig[i * ldc + j];
      expectEq(result, c[i * ldc + j]);
    }
  }
}

TEST(CxxBlasTest, her2kTest) {
  her2kTest< double >();
  her2kTest< float >();
  her2kTest< ::std::complex< double > >();
  her2kTest< ::std::complex< float > >();
}

}  // namespace
}  // namespace linalg
}  // namespace thunder
