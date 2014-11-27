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
#include <cstdlib>

#include "thunder/linalg/blas.hpp"

namespace thunder {
namespace linalg {
namespace cxxblas {

static inline char transChar(Order order, Trans trans) {
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

static inline char uploChar(const Uplo uplo) {
  if (uplo == Uplo::kUpper) {
    return 'U';
  } else if (uplo == Uplo::kLower) {
    return 'L';
  }
  return '\0';
}

template <typename D>
static inline void conj(int n, const D *x, int incx, D *y, int incy) {
  for (int i = 0; i < n; ++i) {
    y[i * incy] = ::std::conj(x[i * incx]);
  }
}

void gbmv(int m, int n, const float *a, const float *x, float *y, float alpha,
          const float beta, int kl, int ku, int lda, int incx, int incy,
          Order order, Trans trans) {
  if (lda == 0) {
    lda = kl + ku + 1;
  }
  char trans_char = transChar(order, trans);
  if (order == Order::kColMajor) {
    sgbmv_(&trans_char, &m, &n, &kl, &ku, &alpha, a, &lda, x, &incx, &beta, y,
           &incy);
  } else {
    sgbmv_(&trans_char, &n, &m, &ku, &kl, &alpha, a, &lda, x, &incx, &beta, y,
           &incy);
  }
}

void gbmv(int m, int n, const double *a, const double *x, double *y,
          double alpha, const double beta, int kl, int ku, int lda, int incx,
          int incy, Order order, Trans trans) {
  if (lda == 0) {
    lda = kl + ku + 1;
  }
  char trans_char = transChar(order, trans);
  if (order == Order::kColMajor) {
    dgbmv_(&trans_char, &m, &n, &kl, &ku, &alpha, a, &lda, x, &incx, &beta, y,
           &incy);
  } else {
    dgbmv_(&trans_char, &n, &m, &ku, &kl, &alpha, a, &lda, x, &incx, &beta, y,
           &incy);
  }
}

void gbmv(int m, int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          ::std::complex< float > alpha, const ::std::complex< float > beta,
          int kl, int ku, int lda, int incx, int incy, Order order,
          Trans trans) {
  if (lda == 0) {
    lda = kl + ku + 1;
  }
  char trans_char = transChar(order, trans);
  if (order == Order::kColMajor) {
    cgbmv_(&trans_char, &m, &n, &kl, &ku, &alpha, a, &lda, x, &incx, &beta, y,
           &incy);
  } else if (trans != Trans::kConjTrans) {
    cgbmv_(&trans_char, &n, &m, &ku, &kl, &alpha, a, &lda, x, &incx, &beta, y,
           &incy);
  } else {
    ::std::complex< float > *xc = nullptr;
    int incxc = 1;
    if (m > 0) {
      xc = static_cast< ::std::complex< float >* >(
          ::std::malloc(m * sizeof(::std::complex< float >)));
      conj(m, x, incx, xc, incxc);
    }
    if (n > 0) {
      conj(n, y, incy, y, incy);
    }
    cgbmv_(&trans_char, &n, &m, &ku, &kl, &alpha, a, &lda, xc, &incxc, &beta, y,
           &incy);
    if (n > 0) {
      conj(n, y, incy, y, incy);
    }
  }
}

void gbmv(int m, int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > alpha, const ::std::complex< double > beta,
          int kl, int ku, int lda, int incx, int incy, Order order,
          Trans trans) {
  if (lda == 0) {
    lda = kl + ku + 1;
  }
  char trans_char = transChar(order, trans);
  if (order == Order::kColMajor) {
    zgbmv_(&trans_char, &m, &n, &kl, &ku, &alpha, a, &lda, x, &incx, &beta, y,
           &incy);
  } else if (trans != Trans::kConjTrans) {
    zgbmv_(&trans_char, &n, &m, &ku, &kl, &alpha, a, &lda, x, &incx, &beta, y,
           &incy);
  } else {
    ::std::complex< double > *xc = nullptr;
    int incxc = 1;
    if (m > 0) {
      xc = static_cast< ::std::complex< double >* >(
          ::std::malloc(m * sizeof(::std::complex< double >)));
      conj(m, x, incx, xc, incxc);
    }
    if (n > 0) {
      conj(n, y, incy, y, incy);
    }
    zgbmv_(&trans_char, &n, &m, &ku, &kl, &alpha, a, &lda, xc, &incxc, &beta, y,
           &incy);
    if (n > 0) {
      conj(n, y, incy, y, incy);
    }
  }
}

void gemv(int m, int n, const float *a, const float *x, float *y, float alpha,
          const float beta, int lda, int incx, int incy, Order order,
          Trans trans) {
  char trans_char = transChar(order, trans);
  if (order == Order::kColMajor) {
    sgemv_(&trans_char, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else {
    sgemv_(&trans_char, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  }
}

void gemv(int m, int n, const double *a, const double *x, double *y,
          double alpha, const double beta, int lda, int incx, int incy,
          Order order, Trans trans) {
  char trans_char = transChar(order, trans);
  if (order == Order::kColMajor) {
    dgemv_(&trans_char, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else {
    dgemv_(&trans_char, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  }
}

void gemv(int m, int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          ::std::complex< float > alpha, const ::std::complex< float > beta,
          int lda, int incx, int incy, Order order, Trans trans) {
  char trans_char = transChar(order, trans);
  if (order == Order::kColMajor) {
    cgemv_(&trans_char, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else {
    cgemv_(&trans_char, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  }
}

void gemv(int m, int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > alpha, const ::std::complex< double > beta,
          int lda, int incx, int incy, Order order, Trans trans) {
  char trans_char = transChar(order, trans);
  if (order == Order::kColMajor) {
    zgemv_(&trans_char, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else {
    zgemv_(&trans_char, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  }
}

void ger(int m, int n, const float *x, const float *y, float *a, float alpha,
         int incx, int incy, int lda, Order order) {
  if (order == Order::kColMajor) {
    sger_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  } else {
    sger_(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
  }
}

void ger(int m, int n, const double *x, const double *y, double *a,
         double alpha, int incx, int incy, int lda, Order order) {
  if (order == Order::kColMajor) {
    dger_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  } else {
    dger_(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
  }
}

void ger(int m, int n, const ::std::complex< float > *x,
         const ::std::complex< float > *y, ::std::complex< float > *a,
         ::std::complex< float > alpha, int incx, int incy, int lda,
         Order order) {
  geru(m, n, x, y, a, alpha, incx, incy, lda, order);
}

void ger(int m, int n, const ::std::complex< double > *x,
         const ::std::complex< double > *y, ::std::complex< double > *a,
         ::std::complex< double > alpha, int incx, int incy, int lda,
         Order order) {
  geru(m, n, x, y, a, alpha, incx, incy, lda, order);
}

void gerc(int m, int n, const float *x, const float *y, float *a, float alpha,
          int incx, int incy, int lda, Order order) {
  ger(m, n, x, y, a, alpha, incx, incy, lda, order);
}

void gerc(int m, int n, const double *x, const double *y, double *a,
          double alpha, int incx, int incy, int lda, Order order) {
  ger(m, n, x, y, a, alpha, incx, incy, lda, order);
}

void gerc(int m, int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *a,
          ::std::complex< float > alpha, int incx, int incy, int lda,
          Order order) {
  if (order == Order::kColMajor) {
    cgerc_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  } else {
    cgerc_(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
  }
}

void gerc(int m, int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *a,
          ::std::complex< double > alpha, int incx, int incy, int lda,
          Order order) {
  if (order == Order::kColMajor) {
    zgerc_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  } else {
    zgerc_(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
  }
}

void geru(int m, int n, const float *x, const float *y, float *a, float alpha,
          int incx, int incy, int lda, Order order) {
  ger(m, n, x, y, a, alpha, incx, incy, lda, order);
}

void geru(int m, int n, const double *x, const double *y, double *a,
          double alpha, int incx, int incy, int lda, Order order) {
  ger(m, n, x, y, a, alpha, incx, incy, lda, order);
}

void geru(int m, int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *a,
          ::std::complex< float > alpha, int incx, int incy, int lda,
          Order order) {
  if (order == Order::kColMajor) {
    cgeru_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  } else {
    cgeru_(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
  }
}

void geru(int m, int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *a,
          ::std::complex< double > alpha, int incx, int incy, int lda,
          Order order) {
  if (order == Order::kColMajor) {
    zgeru_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  } else {
    zgeru_(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
  }
}

void hbmv(int n, const float *a, const float *x, float *y, float alpha,
          const float beta, int k, int lda, int incx, int incy, Order order,
          const Uplo uplo) {
  sbmv(n, a, x, y, alpha, beta, k, lda, incx, incy, order, uplo);
}

void hbmv(int n, const double *a, const double *x, double *y, double alpha,
          const float beta, int k, int lda, int incx, int incy, Order order,
          const Uplo uplo) {
  sbmv(n, a, x, y, alpha, beta, k, lda, incx, incy, order, uplo);
}

void hbmv(int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x,
          ::std::complex< float > *y, ::std::complex< float > alpha,
          const ::std::complex< float > beta, int k, int lda, int incx,
          int incy, Order order, const Uplo uplo) {
  char uplo_char = uploChar(uplo);
  if (order == Order::kRowMajor) {
    chbmv_(&uplo_char, &n, &k, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else {
  }
}

void hbmv(int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > alpha, const ::std::complex< double > beta,
          int k, int lda, int incx, int incy, Order order, const Uplo uplo) {
  // TODO(xiang): complicated implementation
}

}  // namespace cxxblas
}  // namespace linalg
}  // namespace thunder
