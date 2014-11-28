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

#include "thunder/linalg/cxxblas.hpp"

#include <complex>

#include "thunder/linalg/blas.hpp"

namespace thunder {
namespace linalg {
namespace cxxblas {

static inline char transChar(Trans trans) {
  if (trans == Trans::kNoTrans) {
    return 'N';
  } else if (trans == Trans::kTrans) {
    return 'T';
  } else if (trans == Trans::kConjTrans) {
    return 'C';
  }
  return '\0';
}

static inline char uploChar(Order order, Uplo uplo) {
  if (order == Order::kColMajor) {
    if (uplo == Uplo::kUpper) {
      return 'U';
    } else if (uplo == Uplo::kLower) {
      return 'L';
    }
  } else if (order == Order::kRowMajor) {
    if (uplo == Uplo::kUpper) {
      return 'L';
    } else if (uplo == Uplo::kLower) {
      return 'U';
    }
  }
  return '\0';
}

static inline char diagChar(Diag diag) {
  if (diag == Diag::kNonUnit) {
    return 'N';
  } else if (diag == Diag::kUnit){
    return 'U';
  }
  return '\0';
}

template <typename D>
static inline void conj(int n, const D *x, int incx, D *y, int incy) {
  for (int i = 0; i < n; ++i) {
    y[i * incy] = ::std::conj(x[i * incx]);
  }
}

void gemm(int m, int n, int k, const float *a, const float *b, float *c,
          float alpha, float beta, int lda, int ldb, int ldc, Order order,
          Trans transa, Trans transb) {
  char transa_char = transChar(transa);
  char transb_char = transChar(transb);
}

void gemm(int m, int n, int k, const double *a, const double *b, double *c,
          double alpha, double beta, int lda, int ldb, int ldc, Order order,
          Trans transa, Trans transb) {
}

void gemm(int m, int n, int k, const ::std::complex< float > *a,
          const ::std::complex< float > *b, ::std::complex< float > *c,
          ::std::complex< float > alpha, ::std::complex< float > beta, int lda,
          int ldb, int ldc, Order order, Trans transa, Trans transb) {
}

void gemm(int m, int n, int k, const ::std::complex< double > *a,
          const ::std::complex< double > *b, ::std::complex< double > *c,
          ::std::complex< double > alpha, ::std::complex< double > beta,
          int lda, int ldb, int ldc, Order order, Trans transa, Trans transb) {
}

void gemm(int m, int n, int k, const float *a, const ::std::complex< float > *b,
          ::std::complex< float > *c, ::std::complex< float > alpha,
          ::std::complex< float > beta, int lda, int ldb, int ldc, Order order,
          Trans transa, Trans transb) {
}

void gemm(int m, int n, int k, const double *a,
          const ::std::complex< double > *b, ::std::complex< double > *c,
          ::std::complex< double > alpha, ::std::complex< double > beta,
          int lda, int ldb, int ldc, Order order, Trans transa, Trans transb) {
}

}  // namespace cxxblas
}  // namespace linalg
}  // namespace thunder
