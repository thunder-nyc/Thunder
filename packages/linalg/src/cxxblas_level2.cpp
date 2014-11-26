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

static inline char getTransChar(const Order order, const Trans trans) {
  if (order == Order::kColMajor) {
    if (trans == Trans::kNoTrans) {
      return 'N';
    } else if (trans == Trans::kTrans) {
      return 'T';
    } else if (trans == Trans::kConjTrans) {
      return 'C';
    }
  } else if (order == Order::kRowMajor) {
    if (trans == Trans::kNoTrans) {
      return 'T';
    } else if (trans == Trans::kTrans) {
      return 'N';
    } else if (trans == Trans::kConjTrans) {
      return 'N';
    }
  }
  return '\0';
}

static inline char getUploChar(const Uplo uplo) {
  if (uplo == Uplo::kUpper) {
    return 'U';
  } else if (uplo == Uplo::kLower) {
    return 'L';
  }
  return '\0';
}

// Convert banded matrix from row major to column major
template < typename D >
static inline D* rowToColMatrix(int m, int n, D *a, int k,int lda) {
  // TODO(xiang): implement this
}

void gbmv(const int m, const int n, const float *a, const float *x, float *y,
          const float alpha, const float beta, const int kl, const int ku,
          const int lda, const int incx, const int incy, const Order order,
          const Trans trans) {
  // TODO(xiang): Added default argument for lda.
  char trans_char = getTransChar(order, trans);
  if (order == Order::kColMajor) {
    sgbmv_(&trans_char, &m, &n, &ku, &kl, &alpha, a, &lda, x, &incx, &beta, y,
           &incy);
  } else {
    sgbmv_(&trans_char, &n, &m, &ku, &kl, &alpha, a, &lda, x, &incx, &beta, y,
           &incy);
  }
}

void gbmv(const int m, const int n, const double *a, const double *x, double *y,
          const double alpha, const double beta, const int kl, const int ku,
          const int lda, const int incx, const int incy, const Order order,
          const Trans trans) {
  char trans_char = getTransChar(order, trans);
  if (order == Order::kColMajor) {
    dgbmv_(&trans_char, &m, &n, &ku, &kl, &alpha, a, &lda, x, &incx, &beta, y,
           &incy);
  } else {
    dgbmv_(&trans_char, &n, &m, &ku, &kl, &alpha, a, &lda, x, &incx, &beta, y,
           &incy);
  }
}

void gbmv(const int m, const int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          const ::std::complex< float > alpha,
          const ::std::complex< float > beta, const int kl, const int ku,
          const int lda, const int incx, const int incy, const Order order,
          const Trans trans) {
  char trans_char = getTransChar(order, trans);
  if (order == Order::kColMajor) {
    cgbmv_(&trans_char, &m, &n, &ku, &kl, &alpha, a, &lda, x, &incx, &beta, y,
           &incy);
  } else {
    cgbmv_(&trans_char, &n, &m, &ku, &kl, &alpha, a, &lda, x, &incx, &beta, y,
           &incy);
  }
}

void gbmv(const int m, const int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          const ::std::complex< double > alpha,
          const ::std::complex< double > beta, const int kl, const int ku,
          const int lda, const int incx, const int incy, const Order order,
          const Trans trans) {
  char trans_char = getTransChar(order, trans);
  if (order == Order::kColMajor) {
    zgbmv_(&trans_char, &m, &n, &ku, &kl, &alpha, a, &lda, x, &incx, &beta, y,
           &incy);
  } else {
    zgbmv_(&trans_char, &n, &m, &ku, &kl, &alpha, a, &lda, x, &incx, &beta, y,
           &incy);
  }
}

void gemv(const int m, const int n, const float *a, const float *x, float *y,
          const float alpha, const float beta, const int lda, const int incx,
          const int incy, const Order order, const Trans trans) {
  char trans_char = getTransChar(order, trans);
  if (order == Order::kColMajor) {
    sgemv_(&trans_char, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else {
    sgemv_(&trans_char, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  }
}

void gemv(const int m, const int n, const double *a, const double *x, double *y,
          const double alpha, const double beta, const int lda, const int incx,
          const int incy, const Order order, const Trans trans) {
  char trans_char = getTransChar(order, trans);
  if (order == Order::kColMajor) {
    dgemv_(&trans_char, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else {
    dgemv_(&trans_char, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  }
}

void gemv(const int m, const int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          const ::std::complex< float > alpha,
          const ::std::complex< float > beta, const int lda, const int incx,
          const int incy, const Order order, const Trans trans) {
  char trans_char = getTransChar(order, trans);
  if (order == Order::kColMajor) {
    cgemv_(&trans_char, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else {
    cgemv_(&trans_char, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  }
}

void gemv(const int m, const int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          const ::std::complex< double > alpha,
          const ::std::complex< double > beta, const int lda, const int incx,
          const int incy, const Order order, const Trans trans) {
  char trans_char = getTransChar(order, trans);
  if (order == Order::kColMajor) {
    zgemv_(&trans_char, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else {
    zgemv_(&trans_char, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  }
}

void gemv(const int m, const int n, float *a, const ::std::complex< float > *x,
          ::std::complex< float > *y, const ::std::complex< float > alpha,
          const ::std::complex< float > beta, const int lda, const int incx,
          const int incy, const Order order, const Trans trans) {
  char trans_char = getTransChar(order, trans);
  if (order == Order::kColMajor) {
    scgemv_(&trans_char, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else {
    scgemv_(&trans_char, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  }
}

void gemv(const int m, const int n, double *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          const ::std::complex< double > alpha,
          const ::std::complex< double > beta, const int lda, const int incx,
          const int incy, const Order order, const Trans trans) {
  char trans_char = getTransChar(order, trans);
  if (order == Order::kColMajor) {
    dzgemv_(&trans_char, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else {
    dzgemv_(&trans_char, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  }
}

void ger(const int m, const int n, const float *x, const float *y, float *a,
         const float alpha, const int incx, const int incy, const int lda,
         const Order order) {
  if (order == Order::kColMajor) {
    sger_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  } else {
    sger_(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
  }
}

void ger(const int m, const int n, const double *x, const double *y, double *a,
         const double alpha, const int incx, const int incy, const int lda,
         const Order order) {
  if (order == Order::kColMajor) {
    dger_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  } else {
    dger_(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
  }
}

void ger(const int m, const int n, const ::std::complex< float > *x,
         const ::std::complex< float > *y, ::std::complex< float > *a,
         const ::std::complex< float > alpha, const int incx, const int incy,
         const int lda, const Order order) {
  cgeru(m, n, x, y, a, alpha, incx, incy, lda, order);
}

void ger(const int m, const int n, const ::std::complex< double > *x,
         const ::std::complex< double > *y, ::std::complex< double > *a,
         const ::std::complex< double > alpha, const int incx, const int incy,
         const int lda, const Order order) {
  zgeru(m, n, x, y, a, alpha, incx, incy, lda, order);
}

void gerc(const int m, const int n, const float *x, const float *y, float *a,
          const float alpha, const int incx, const int incy, const int lda,
          const Order order) {
  ger(m, n, x, y, a, alpha, incx, incy, lda, order);
}

void gerc(const int m, const int n, const double *x, const double *y, double *a,
          const double alpha, const int incx, const int incy, const int lda,
          const Order order) {
  ger(m, n, x, y, a, alpha, incx, incy, lda, order);
}

void gerc(const int m, const int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *a,
          const ::std::complex< float > alpha, const int incx, const int incy,
          const int lda, const Order order) {
  if (order == Order::kColMajor) {
    cgerc_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  } else {
    cgerc_(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
  }
}

void gerc(const int m, const int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *a,
          const ::std::complex< double > alpha, const int incx, const int incy,
          const int lda, const Order order) {
  if (order == Order::kColMajor) {
    zgerc_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  } else {
    zgerc_(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
  }
}

void geru(const int m, const int n, const float *x, const float *y, float *a,
          const float alpha, const int incx, const int incy, const int lda,
          const Order order) {
  ger(m, n, x, y, a, alpha, incx, incy, lda, order);
}

void geru(const int m, const int n, const double *x, const double *y, double *a,
          const double alpha, const int incx, const int incy, const int lda,
          const Order order) {
  ger(m, n, x, y, a, alpha, incx, incy, lda, order);
}

void geru(const int m, const int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *a,
          const ::std::complex< float > alpha, const int incx, const int incy,
          const int lda, const Order order) {
  if (order == Order::kColMajor) {
    cgeru_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  } else {
    cgeru_(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
  }
}

void geru(const int m, const int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *a,
          const ::std::complex< double > alpha, const int incx, const int incy,
          const int lda, const Order order) {
  if (order == Order::kColMajor) {
    zgeru_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  } else {
    zgeru_(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
  }
}

void hbmv(const int n, const float *a, const float *x, float *y,
          const float alpha, const float beta, const int k, const int lda,
          const int incx, const int incy, const Order order, const Uplo uplo) {
  sbmv(n, a, x, y, alpha, beta, k, lda, incx, incy, order, uplo);
}

void hbmv(const int n, const double *a, const double *x, double *y,
          const double alpha, const float beta, const int k, const int lda,
          const int incx, const int incy, const Order order, const Uplo uplo) {
  sbmv(n, a, x, y, alpha, beta, k, lda, incx, incy, order, uplo);
}

void hbmv(const int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          const ::std::complex< float > alpha,
          const ::std::complex< float > beta, const int k, const int lda,
          const int incx, const int incy, const Order order, const Uplo uplo) {
  char uplo_char = getUploChar(uplo);
  if (order == Order::kRowMajor) {
    chbmv_(&uplo_char, &n, &k, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else {
    
  }
}

void hbmv(const int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          const ::std::complex< double > alpha,
          const ::std::complex< double > beta, const int k, const int lda,
          const int incx, const int incy, const Order order, const Uplo uplo) {
  // TODO(xiang): complicated implementation
}

}  // namespace cxxblas
}  // namespace linalg
}  // namespace thunder
