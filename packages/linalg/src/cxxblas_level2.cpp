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

#include <algorithm>
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
    }
    conj(m, x, incx, xc, incxc);
    conj(n, y, incy, y, incy);
    cgbmv_(&trans_char, &n, &m, &ku, &kl, &alpha, a, &lda, xc, &incxc, &beta, y,
           &incy);
    conj(n, y, incy, y, incy);
    if (m > 0) {
      ::std::free(xc);
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
    }
    conj(m, x, incx, xc, incxc);
    conj(n, y, incy, y, incy);
    zgbmv_(&trans_char, &n, &m, &ku, &kl, &alpha, a, &lda, xc, &incxc, &beta, y,
           &incy);
    conj(n, y, incy, y, incy);
    if (m > 0) {
      ::std::free(xc);
    }
  }
}

void gemv(int m, int n, const float *a, const float *x, float *y, float alpha,
          const float beta, int lda, int incx, int incy, Order order,
          Trans trans) {
  if (lda == 0) {
    lda = ::std::max(1, m);
  }
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
  if (lda == 0) {
    if (order == Order::kColMajor) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
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
  if (lda == 0) {
    if (order == Order::kColMajor) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
  char trans_char = transChar(order, trans);
  if (order == Order::kColMajor) {
    cgemv_(&trans_char, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else if (trans != Trans::kConjTrans) {
    cgemv_(&trans_char, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else {
    ::std::complex< float > *xc = nullptr;
    int incxc = 1;
    if (m > 0) {
      xc = static_cast< ::std::complex< float >* >(
          ::std::malloc(m * sizeof(::std::complex< float >)));
    }
    conj(m, x, incx, xc, incxc);
    conj(n, y, incy, y, incy);
    cgemv_(&trans_char, &n, &m, &alpha, a, &lda, xc, &incxc, &beta, y, &incy);
    conj(n, y, incy, y, incy);
    if (m > 0) {
      ::std::free(xc);
    }
  }
}

void gemv(int m, int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > alpha, const ::std::complex< double > beta,
          int lda, int incx, int incy, Order order, Trans trans) {
  if (lda == 0) {
    if (order == Order::kColMajor) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
  char trans_char = transChar(order, trans);
  if (order == Order::kColMajor) {
    zgemv_(&trans_char, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else if (trans != Trans::kConjTrans) {
    zgemv_(&trans_char, &n, &m, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else {
    ::std::complex< double > *xc = nullptr;
    int incxc = 1;
    if (m > 0) {
      xc = static_cast< ::std::complex< double >* >(
          ::std::malloc(m * sizeof(::std::complex< double >)));
    }
    conj(m, x, incx, xc, incxc);
    conj(n, y, incy, y, incy);
    zgemv_(&trans_char, &n, &m, &alpha, a, &lda, xc, &incxc, &beta, y, &incy);
    conj(n, y, incy, y, incy);
    if (m > 0) {
      ::std::free(xc);
    }
  }
}

void ger(int m, int n, const float *x, const float *y, float *a, float alpha,
         int incx, int incy, int lda, Order order) {
  if (lda == 0) {
    if (order == Order::kColMajor) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
  if (order == Order::kColMajor) {
    sger_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  } else {
    sger_(&n, &m, &alpha, y, &incy, x, &incx, a, &lda);
  }
}

void ger(int m, int n, const double *x, const double *y, double *a,
         double alpha, int incx, int incy, int lda, Order order) {
  if (lda == 0) {
    if (order == Order::kColMajor) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
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
  if (lda == 0) {
    if (order == Order::kColMajor) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
  if (order == Order::kColMajor) {
    cgerc_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  } else {
    ::std::complex< float > *yc = nullptr;
    int incyc = 1;
    if (n > 0) {
      yc = static_cast< ::std::complex< float >* >(
          ::std::malloc(n * sizeof(::std::complex< float >)));
    }
    conj(n, y, incy, yc, incyc);
    cgeru_(&n, &m, &alpha, yc, &incyc, x, &incx, a, &lda);
    if (n > 0) {
      ::std::free(yc);
    }
  }
}

void gerc(int m, int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *a,
          ::std::complex< double > alpha, int incx, int incy, int lda,
          Order order) {
  if (lda == 0) {
    if (order == Order::kColMajor) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
  if (order == Order::kColMajor) {
    zgerc_(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
  } else {
    ::std::complex< double > *yc = nullptr;
    int incyc = 1;
    if (n > 0) {
      yc = static_cast< ::std::complex< double >* >(
          ::std::malloc(n * sizeof(::std::complex< double >)));
    }
    conj(n, y, incy, yc, incyc);
    zgeru_(&n, &m, &alpha, yc, &incyc, x, &incx, a, &lda);
    if (n > 0) {
      ::std::free(yc);
    }
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
  if (lda == 0) {
    if (order == Order::kColMajor) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
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
  if (lda == 0) {
    if (order == Order::kColMajor) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
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
  if (lda == 0) {
    lda = k + 1;
  }
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    chbmv_(&uplo_char, &n, &k, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else {
    ::std::complex< float > *xc = nullptr;
    int incxc = 1;
    if (n > 0) {
      xc = static_cast< ::std::complex< float >* >(
          ::std::malloc(n * sizeof(::std::complex< float >)));
    }
    conj(n, x, incx, xc, incxc);
    conj(n, y, incy, y, incy);
    chbmv_(&uplo_char, &n, &k, &alpha, a, &lda, xc, &incxc, &beta, y, &incy);
    conj(n, y, incy, y, incy);
    if (n > 0) {
      ::std::free(xc);
    }
  }
}

void hbmv(int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > alpha, const ::std::complex< double > beta,
          int k, int lda, int incx, int incy, Order order, const Uplo uplo) {
  if (lda == 0) {
    lda = k + 1;
  }
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    zhbmv_(&uplo_char, &n, &k, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else {
    ::std::complex< double > *xc = nullptr;
    int incxc = 1;
    if (n > 0) {
      xc = static_cast< ::std::complex< double >* >(
          ::std::malloc(n * sizeof(::std::complex< double >)));
    }
    conj(n, x, incx, xc, incxc);
    conj(n, y, incy, y, incy);
    zhbmv_(&uplo_char, &n, &k, &alpha, a, &lda, xc, &incxc, &beta, y, &incy);
    conj(n, y, incy, y, incy);
    if (n > 0) {
      ::std::free(xc);
    }
  }
}

void hemv(int n, const float *a, const float *x, float *y, float alpha,
          float beta, int lda, int incx, int incy, Order order, Uplo uplo) {
  symv(n, a, x, y, alpha, beta, lda, incx, incy, order, uplo);
}

void hemv(int n, const double *a, const double *x, double *y, double alpha,
          float beta, int lda, int incx, int incy, Order order, Uplo uplo) {
  symv(n, a, x, y, alpha, beta, lda, incx, incy, order, uplo);
}

void hemv(int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          ::std::complex< float > alpha, ::std::complex< float > beta, int lda,
          int incx, int incy, Order order, Uplo uplo) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    chemv_(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else {
    ::std::complex< float > *xc = nullptr;
    int incxc = 1;
    if (n > 0) {
      xc = static_cast< ::std::complex< float >* >(
          ::std::malloc(n * sizeof(::std::complex< float >)));
    }
    conj(n, x, incx, xc, incxc);
    conj(n, y, incy, y, incy);
    chemv_(&uplo_char, &n, &alpha, a, &lda, xc, &incxc, &beta, y, &incy);
    conj(n, y, incy, y, incy);
    if (n > 0) {
      ::std::free(xc);
    }
  }
}

void hemv(int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > alpha, ::std::complex< double > beta,
          int lda, int incx, int incy, Order order, Uplo uplo) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    zhemv_(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
  } else {
    ::std::complex< double > *xc = nullptr;
    int incxc = 1;
    if (n > 0) {
      xc = static_cast< ::std::complex< double >* >(
          ::std::malloc(n * sizeof(::std::complex< double >)));
    }
    conj(n, x, incx, xc, incxc);
    conj(n, y, incy, y, incy);
    zhemv_(&uplo_char, &n, &alpha, a, &lda, xc, &incxc, &beta, y, &incy);
    conj(n, y, incy, y, incy);
    if (n > 0) {
      ::std::free(xc);
    }
  }
}

void her(int n, const float *x, float *a, float alpha, int incx, int lda,
         Order order, Uplo uplo) {
  syr(n, x, a, alpha, incx, lda, order, uplo);
}

void her(int n, const double *x, double *a, double alpha, int incx, int lda,
         Order order, Uplo uplo) {
  syr(n, x, a, alpha, incx, lda, order, uplo);
}

void her(int n, const ::std::complex< float > *x, ::std::complex< float > *a,
         ::std::complex< float > alpha, int incx, int lda, Order order,
         Uplo uplo) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    cher_(&uplo_char, &n, &alpha, x, &incx, a, &lda);
  } else {
    ::std::complex< float > *xc = nullptr;
    int incxc = 1;
    if (n > 0) {
      xc = static_cast< ::std::complex< float >* >(
          ::std::malloc(n * sizeof(::std::complex< float >)));
    }
    conj(n, x, incx, xc, incxc);
    cher_(&uplo_char, &n, &alpha, xc, &incxc, a, &lda);
    if (n > 0) {
      ::std::free(xc);
    }
  }
}

void her(int n, const ::std::complex< double > *x, ::std::complex< double > *a,
         ::std::complex< double > alpha, int incx, int lda, Order order,
         Uplo uplo) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    zher_(&uplo_char, &n, &alpha, x, &incx, a, &lda);
  } else {
    ::std::complex< double > *xc = nullptr;
    int incxc = 1;
    if (n > 0) {
      xc = static_cast< ::std::complex< double >* >(
          ::std::malloc(n * sizeof(::std::complex< double >)));
    }
    conj(n, x, incx, xc, incxc);
    zher_(&uplo_char, &n, &alpha, xc, &incxc, a, &lda);
    if (n > 0) {
      ::std::free(xc);
    }
  }
}

void her2(int n, const float *x, const float *y, float *a, float alpha,
          int incx, int incy, int lda, Order order, Uplo uplo) {
  syr2(n, x, y, a, alpha, incx, incy, lda, order, uplo);
}

void her2(int n, const double *x, const double *y, double *a, double alpha,
          int incx, int incy, int lda, Order order, Uplo uplo) {
  syr2(n, x, y, a, alpha, incx, incy, lda, order, uplo);
}

void her2(int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *a,
          ::std::complex< float > alpha, int incx, int incy, int lda,
          Order order, Uplo uplo) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    cher2_(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
  } else {
    ::std::complex< float > *xc = nullptr;
    int incxc = 1;
    ::std::complex< float > *yc = nullptr;
    int incyc = 1;
    if (n > 0) {
      xc = static_cast< ::std::complex< float >* >(
          ::std::malloc(n * sizeof(::std::complex< double >)));
      yc = static_cast< ::std::complex< float >* >(
          ::std::malloc(n * sizeof(::std::complex< double >)));
    }
    conj(n, x, incx, xc, incxc);
    conj(n, y, incy, yc, incyc);
    cher2_(&uplo_char, &n, &alpha, xc, &incxc, yc, &incyc, a, &lda);
    if (n > 0) {
      ::std::free(xc);
      ::std::free(yc);
    }
  }
}

void her2(int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *a,
          ::std::complex< double > alpha, int incx, int incy, int lda,
          Order order, Uplo uplo) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    zher2_(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
  } else {
    ::std::complex< double > *xc = nullptr;
    int incxc = 1;
    ::std::complex< double > *yc = nullptr;
    int incyc = 1;
    if (n > 0) {
      xc = static_cast< ::std::complex< double >* >(
          ::std::malloc(n * sizeof(::std::complex< double >)));
      yc = static_cast< ::std::complex< double >* >(
          ::std::malloc(n * sizeof(::std::complex< double >)));
    }
    conj(n, x, incx, xc, incxc);
    conj(n, y, incy, yc, incyc);
    zher2_(&uplo_char, &n, &alpha, xc, &incxc, yc, &incyc, a, &lda);
    if (n > 0) {
      ::std::free(xc);
      ::std::free(yc);
    }
  }
}

void hpmv(int n, const float *ap, const float *x, float *y, float alpha,
          float beta, int incx, int incy, Order order, Uplo uplo) {
  spmv(n, ap, x, y, alpha, beta, incx, incy, order, uplo);
}

void hpmv(int n, const double *ap, const double *x, double *y, double alpha,
          double beta, int incx, int incy, Order order, Uplo uplo) {
  spmv(n, ap, x, y, alpha, beta, incx, incy, order, uplo);
}

void hpmv(int n, const ::std::complex< float > *ap,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          ::std::complex< float > alpha, ::std::complex< float > beta, int incx,
          int incy, Order order, Uplo uplo) {
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    chpmv_(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
  } else {
    ::std::complex< float > *xc = nullptr;
    int incxc = 1;
    if (n > 0) {
      xc = static_cast< ::std::complex< float >* >(
          ::std::malloc(n * sizeof(::std::complex< float >)));
    }
    conj(n, x, incx, xc, incxc);
    conj(n, y, incy, y, incy);
    chpmv_(&uplo_char, &n, &alpha, ap, xc, &incxc, &beta, y, &incy);
    conj(n, y, incy, y, incy);
    if (n > 0) {
      ::std::free(xc);
    }
  }
}

void hpmv(int n, const ::std::complex< double > *ap,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > alpha, ::std::complex< double > beta,
          int incx, int incy, Order order, Uplo uplo) {
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    zhpmv_(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
  } else {
    ::std::complex< double > *xc = nullptr;
    int incxc = 1;
    if (n > 0) {
      xc = static_cast< ::std::complex< double >* >(
          ::std::malloc(n * sizeof(::std::complex< double >)));
    }
    conj(n, x, incx, xc, incxc);
    conj(n, y, incy, y, incy);
    zhpmv_(&uplo_char, &n, &alpha, ap, xc, &incxc, &beta, y, &incy);
    conj(n, y, incy, y, incy);
    if (n > 0) {
      ::std::free(xc);
    }
  }
}

void hpr(int n, const float *x, float *ap, float alpha, int incx, Order order,
         Uplo uplo) {
  spr(n, x, ap, alpha, incx, order, uplo);
}

void hpr(int n, const double *x, double *ap, double alpha, int incx,
         Order order, Uplo uplo) {
  spr(n, x, ap, alpha, incx, order, uplo);
}

void hpr(int n, const ::std::complex< float > *x, ::std::complex< float > *ap,
         ::std::complex< float > alpha, int incx, Order order, Uplo uplo) {
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    chpr_(&uplo_char, &n, &alpha, x, &incx, ap);
  } else {
    ::std::complex< float > *xc = nullptr;
    int incxc = 1;
    if (n > 0) {
      xc = static_cast< ::std::complex< float >* >(
          ::std::malloc(n * sizeof(::std::complex< float >)));
    }
    conj(n, x, incx, xc, incxc);
    chpr_(&uplo_char, &n, &alpha, xc, &incxc, ap);
    if (n > 0) {
      ::std::free(xc);
    }
  }
}

void hpr(int n, const ::std::complex< double > *x, ::std::complex< double > *ap,
         ::std::complex< double > alpha, int incx, Order order, Uplo uplo) {
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    zhpr_(&uplo_char, &n, &alpha, x, &incx, ap);
  } else {
    ::std::complex< double > *xc = nullptr;
    int incxc = 1;
    if (n > 0) {
      xc = static_cast< ::std::complex< double >* >(
          ::std::malloc(n * sizeof(::std::complex< double >)));
    }
    conj(n, x, incx, xc, incxc);
    zhpr_(&uplo_char, &n, &alpha, xc, &incxc, ap);
    if (n > 0) {
      ::std::free(xc);
    }
  }
}

void hpr2(int n, const float *x, const float *y, float *ap, float alpha,
          int incx, int incy, Order order, Uplo uplo) {
  spr2(n, x, y, ap, alpha, incx, incy, order, uplo);
}

void hpr2(int n, const double *x, const double *y, double *ap, double alpha,
          int incx, int incy, Order order, Uplo uplo) {
  spr2(n, x, y, ap, alpha, incx, incy, order, uplo);
}

void hpr2(int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *ap,
          ::std::complex< float > alpha, int incx, int incy, Order order,
          Uplo uplo) {
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    chpr2_(&uplo_char, &n, &alpha, x, &incx, y, &incy, ap);
  } else {
    ::std::complex< float > *xc = nullptr;
    int incxc = 1;
    ::std::complex< float > *yc = nullptr;
    int incyc = 1;
    if (n > 0) {
      xc = static_cast< ::std::complex< float >* >(
          ::std::malloc(n * sizeof(::std::complex< float >)));
      yc = static_cast< ::std::complex< float >* >(
          ::std::malloc(n * sizeof(::std::complex< float >)));
    }
    conj(n, x, incx, xc, incxc);
    conj(n, y, incy, yc, incyc);
    chpr2_(&uplo_char, &n, &alpha, xc, &incxc, yc, &incyc, ap);
    if (n > 0) {
      ::std::free(xc);
      ::std::free(yc);
    }
  }
}

void hpr2(int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *ap,
          ::std::complex< double > alpha, int incx, int incy, Order order,
          Uplo uplo) {
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    zhpr2_(&uplo_char, &n, &alpha, x, &incx, y, &incy, ap);
  } else {
    ::std::complex< double > *xc = nullptr;
    int incxc = 1;
    ::std::complex< double > *yc = nullptr;
    int incyc = 1;
    if (n > 0) {
      xc = static_cast< ::std::complex< double >* >(
          ::std::malloc(n * sizeof(::std::complex< double >)));
      yc = static_cast< ::std::complex< double >* >(
          ::std::malloc(n * sizeof(::std::complex< double >)));
    }
    conj(n, x, incx, xc, incxc);
    conj(n, y, incy, yc, incyc);
    zhpr2_(&uplo_char, &n, &alpha, xc, &incxc, yc, &incyc, ap);
    if (n > 0) {
      ::std::free(xc);
      ::std::free(yc);
    }
  }
}

void sbmv(int n, const float *a, const float *x, float *y, float alpha,
          float beta, int k, int lda, int incx, int incy, Order order,
          Uplo uplo) {
  if (lda == 0) {
    lda = k + 1;
  }
  char uplo_char = uploChar(order, uplo);
  ssbmv_(&uplo_char, &n, &k, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void sbmv(int n, const double *a, const double *x, double *y, double alpha,
          double beta, int k, int lda, int incx, int incy, Order order,
          Uplo uplo) {
  if (lda == 0) {
    lda = k + 1;
  }
  char uplo_char = uploChar(order, uplo);
  dsbmv_(&uplo_char, &n, &k, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void sbmv(int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          ::std::complex< float > alpha, ::std::complex< float > beta, int k,
          int lda, int incx, int incy, Order order, Uplo uplo) {
  hbmv(n, a, x, y, alpha, beta, k, lda, incx, incy, order, uplo);
}

void sbmv(int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > alpha, ::std::complex< double > beta, int k,
          int lda, int incx, int incy, Order order, Uplo uplo) {
  hbmv(n, a, x, y, alpha, beta, k, lda, incx, incy, order, uplo);
}

void spmv(int n, const float *ap, const float *x, float *y, float alpha,
          float beta, int incx, int incy, Order order, Uplo uplo) {
  char uplo_char = uploChar(order, uplo);
  sspmv_(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
}

void spmv(int n, const double *ap, const double *x, double *y, double alpha,
          double beta, int incx, int incy, Order order, Uplo uplo) {
  char uplo_char = uploChar(order, uplo);
  dspmv_(&uplo_char, &n, &alpha, ap, x, &incx, &beta, y, &incy);
}

void spmv(int n, const ::std::complex< float > *ap,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          ::std::complex< float > alpha, ::std::complex< float > beta, int incx,
          int incy, Order order, Uplo uplo) {
  hpmv(n, ap, x, y, alpha, beta, incx, incy, order, uplo);
}

void spmv(int n, const ::std::complex< double > *ap,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > alpha, ::std::complex< double > beta,
          int incx, int incy, Order order, Uplo uplo) {
  hpmv(n, ap, x, y, alpha, beta, incx, incy, order, uplo);
}

void spr(int n, const float *x, float *ap, float alpha, int incx,Order order,
         Uplo uplo) {
  char uplo_char = uploChar(order, uplo);
  sspr_(&uplo_char, &n, &alpha, x, &incx, ap);
}

void spr(int n, const double *x, double *ap, double alpha, int incx,
         Order order, Uplo uplo) {
  char uplo_char = uploChar(order, uplo);
  dspr_(&uplo_char, &n, &alpha, x, &incx, ap);
}

void spr(int n, const ::std::complex< float > *x, ::std::complex< float > *ap,
         ::std::complex< float > alpha, int incx, Order order, Uplo uplo) {
  hpr(n, x, ap, alpha, incx, order, uplo);
}

void spr(int n, const ::std::complex< double > *x, ::std::complex< double > *ap,
         ::std::complex< double > alpha, int incx, Order order, Uplo uplo) {
  hpr(n, x, ap, alpha, incx, order, uplo);
}

void spr2(int n, const float *x, const float *y, float *ap, float alpha,
          int incx, int incy, Order order, Uplo uplo) {
  char uplo_char = uploChar(order, uplo);
  sspr2_(&uplo_char, &n, &alpha, x, &incx, y, &incy, ap);
}

void spr2(int n, const double *x, const double *y, double *ap, double alpha,
          int incx, int incy, Order order, Uplo uplo) {
  char uplo_char = uploChar(order, uplo);
  dspr2_(&uplo_char, &n, &alpha, x, &incx, y, &incy, ap);
}

void spr2(int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *ap,
          ::std::complex< float > alpha, int incx, int incy, Order order,
          Uplo uplo) {
  hpr2(n, x, y, ap, alpha, incx, incy, order, uplo);
}

void spr2(int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *ap,
          ::std::complex< double > alpha, int incx, int incy,  Order order,
          Uplo uplo) {
  hpr2(n, x, y, ap, alpha, incx, incy, order, uplo);
}

void symv(int n, const float *a, const float *x, float *y, float alpha,
          float beta, int lda, int incx, int incy, Order order, Uplo uplo) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  ssymv_(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void symv(int n, const double *a, const double *x, double *y, double alpha,
          double beta, int lda, int incx, int incy, Order order, Uplo uplo) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  dsymv_(&uplo_char, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

void symv(int n, const ::std::complex< float > *a,
          const ::std::complex< float > *x, ::std::complex< float > *y,
          ::std::complex< float > alpha, ::std::complex< float > beta, int lda,
          int incx, int incy, Order order, Uplo uplo) {
  hemv(n, a, x, y, alpha, beta, lda, incx, incy, order, uplo);
}

void symv(int n, const ::std::complex< double > *a,
          const ::std::complex< double > *x, ::std::complex< double > *y,
          ::std::complex< double > alpha, ::std::complex< double > beta,
          int lda, int incx, int incy, Order order, Uplo uplo) {
  hemv(n, a, x, y, alpha, beta, lda, incx, incy, order, uplo);
}

void syr(int n, const float *x, float *a, float alpha, int incx, int lda,
         Order order, Uplo uplo) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  ssyr_(&uplo_char, &n, &alpha, x, &incx, a, &lda);
}

void syr(int n, const double *x, double *a, double alpha, int incx, int lda,
         Order order, Uplo uplo) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  dsyr_(&uplo_char, &n, &alpha, x, &incx, a, &lda);
}

void syr(int n, const ::std::complex< float > *x, ::std::complex< float > *a,
         ::std::complex< float > alpha, int incx, int lda, Order order,
         Uplo uplo) {
  her(n, x, a, alpha, incx, lda, order, uplo);
}

void syr(int n, const ::std::complex< double > *x, ::std::complex< double > *a,
         ::std::complex< double > alpha, int incx, int lda, Order order,
         Uplo uplo) {
  her(n, x, a, alpha, incx, lda, order, uplo);
}

void syr2(int n, const float *x, const float *y, float *a, float alpha,
          int incx, int incy, int lda, Order order, Uplo uplo) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  ssyr2_(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
}

void syr2(int n, const double *x, const double *y, double *a, double alpha,
          int incx, int incy, int lda, Order order, Uplo uplo) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  dsyr2_(&uplo_char, &n, &alpha, x, &incx, y, &incy, a, &lda);
}

void syr2(int n, const ::std::complex< float > *x,
          const ::std::complex< float > *y, ::std::complex< float > *a,
          ::std::complex< float > alpha, int incx, int incy,  int lda,
          Order order, Uplo uplo) {
  her2(n, x, y, a, alpha, incx, incy, lda, order, uplo);
}

void syr2(int n, const ::std::complex< double > *x,
          const ::std::complex< double > *y, ::std::complex< double > *a,
          ::std::complex< double > alpha, int incx, int incy, int lda,
          Order order, Uplo uplo) {
  her2(n, x, y, a, alpha, incx, incy, lda, order, uplo);
}

void tbmv(int n, const float *a, float *x, int k, int lda, int incx,
          Order order, Uplo uplo, Trans trans, Diag diag) {
  if (lda == 0) {
    lda = k + 1;
  }
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  stbmv_(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
}

void tbmv(int n, const double *a, double *x, int k, int lda, int incx,
          Order order, Uplo uplo, Trans trans, Diag diag) {
  if (lda == 0) {
    lda = k + 1;
  }
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  dtbmv_(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
}

void tbmv(int n, const ::std::complex< float > *a, ::std::complex< float > *x,
          int k, int lda, int incx,  Order order, Uplo uplo, Trans trans,
          Diag diag) {
  if (lda == 0) {
    lda = k + 1;
  }
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor || trans != Trans::kConjTrans) {
    ctbmv_(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
  } else {
    conj(n, x, incx, x, incx);
    ctbmv_(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
    conj(n, x, incx, x, incx);
  }
}

void tbmv(int n, const ::std::complex< double > *a, ::std::complex< double > *x,
          int k, int lda, int incx, Order order, Uplo uplo, Trans trans,
          Diag diag) {
  if (lda == 0) {
    lda = k + 1;
  }
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor || trans != Trans::kConjTrans) {
    ztbmv_(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
  } else {
    conj(n, x, incx, x, incx);
    ztbmv_(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
    conj(n, x, incx, x, incx);
  }
}

void tbsv(int n, const float *a, float *x, int k, int lda, int incx,
          Order order, Uplo uplo, Trans trans, Diag diag) {
  if (lda == 0) {
    lda = k + 1;
  }
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  stbsv_(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
}

void tbsv(int n, const double *a, double *x, int k, int lda, int incx,
          Order order, Uplo uplo, Trans trans, Diag diag) {
  if (lda == 0) {
    lda = k + 1;
  }
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  dtbsv_(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
}

void tbsv(int n, const ::std::complex< float > *a, ::std::complex< float > *x,
          int k, int lda, int incx,  Order order, Uplo uplo, Trans trans,
          Diag diag) {
  if (lda == 0) {
    lda = k + 1;
  }
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor || trans != Trans::kConjTrans) {
    ctbsv_(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
  } else {
    conj(n, x, incx, x, incx);
    ctbsv_(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
    conj(n, x, incx, x, incx);
  }
}

void tbsv(int n, const ::std::complex< double > *a, ::std::complex< double > *x,
          int k, int lda, int incx, Order order, Uplo uplo, Trans trans,
          Diag diag) {
  if (lda == 0) {
    lda = k + 1;
  }
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor || trans != Trans::kConjTrans) {
    ztbsv_(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
  } else {
    conj(n, x, incx, x, incx);
    ztbsv_(&uplo_char, &trans_char, &diag_char, &n, &k, a, &lda, x, &incx);
    conj(n, x, incx, x, incx);
  }
}

void tpmv(int n, const float *ap, float *x, int incx, Order order, Uplo uplo,
          Trans trans, Diag diag) {
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  stpmv_(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);  
}

void tpmv(int n, const double *ap, double *x, int incx, Order order, Uplo uplo,
          Trans trans, Diag diag) {
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  dtpmv_(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
}

void tpmv(int n, const ::std::complex< float > *ap, ::std::complex< float > *x,
          int incx, Order order, Uplo uplo, Trans trans, Diag diag) {
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor || trans != Trans::kConjTrans) {
    ctpmv_(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
  } else {
    conj(n, x, incx, x, incx);
    ctpmv_(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
    conj(n, x, incx, x, incx);
  }
}

void tpmv(int n, const ::std::complex< double > *ap,
          ::std::complex< double > *x, int incx, Order order, Uplo uplo,
          Trans trans, Diag diag) {
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor || trans != Trans::kConjTrans) {
    ztpmv_(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
  } else {
    conj(n, x, incx, x, incx);
    ztpmv_(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
    conj(n, x, incx, x, incx);
  }
}

void tpsv(int n, const float *ap, float *x, int incx, Order order, Uplo uplo,
          Trans trans, Diag diag) {
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  stpsv_(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);  
}

void tpsv(int n, const double *ap, double *x, int incx, Order order, Uplo uplo,
          Trans trans, Diag diag) {
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  dtpsv_(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);  
}

void tpsv(int n, const ::std::complex< float > *ap, ::std::complex< float > *x,
          int incx, Order order, Uplo uplo, Trans trans, Diag diag) {
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor || trans != Trans::kConjTrans) {
    ctpsv_(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
  } else {
    conj(n, x, incx, x, incx);
    ctpsv_(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
    conj(n, x, incx, x, incx);
  }
}

void tpsv(int n, const ::std::complex< double > *ap,
          ::std::complex< double > *x, int incx, Order order, Uplo uplo,
          Trans trans, Diag diag) {
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor || trans != Trans::kConjTrans) {
    ztpsv_(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
  } else {
    conj(n, x, incx, x, incx);
    ztpsv_(&uplo_char, &trans_char, &diag_char, &n, ap, x, &incx);
    conj(n, x, incx, x, incx);
  }
}

void trmv(int n, const float *a, float *x, int lda, int incx, Order order,
          Uplo uplo, Trans trans, Diag diag) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  strmv_(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
}

void trmv(int n, const double *a, double *x, int lda, int incx, Order order,
          Uplo uplo, Trans trans, Diag diag) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  dtrmv_(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
}

void trmv(int n, const ::std::complex< float > *a, ::std::complex< float > *x,
          int lda, int incx, Order order, Uplo uplo, Trans trans, Diag diag) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor || trans != Trans::kConjTrans) {
    ctrmv_(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
  } else {
    conj(n, x, incx, x, incx);
    ctrmv_(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
    conj(n, x, incx, x, incx);
  }
}

void trmv(int n, const ::std::complex< double > *a, ::std::complex< double > *x,
          int lda, int incx, Order order, Uplo uplo, Trans trans, Diag diag) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor || trans != Trans::kConjTrans) {
    ztrmv_(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
  } else {
    conj(n, x, incx, x, incx);
    ztrmv_(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
    conj(n, x, incx, x, incx);
  }
}

void trsv(int n, const float *a, float *x, int lda, int incx, Order order,
          Uplo uplo, Trans trans, Diag diag) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  strsv_(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
}

void trsv(int n, const double *a, double *x, int lda, int incx, Order order,
          Uplo uplo, Trans trans, Diag diag) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  dtrsv_(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
}

void trsv(int n, const ::std::complex< float > *a, ::std::complex< float > *x,
          int lda, int incx, Order order, Uplo uplo, Trans trans, Diag diag) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor || trans != Trans::kConjTrans) {
    ctrsv_(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
  } else {
    conj(n, x, incx, x, incx);
    ctrsv_(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
    conj(n, x, incx, x, incx);
  }
}

void trsv(int n, const ::std::complex< double > *a, ::std::complex< double > *x,
          int lda, int incx, Order order, Uplo uplo, Trans trans, Diag diag) {
  if (lda == 0) {
    lda = ::std::max(1, n);
  }
  char trans_char = transChar(order, trans);
  char uplo_char = uploChar(order, uplo);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor || trans != Trans::kConjTrans) {
    ztrsv_(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
  } else {
    conj(n, x, incx, x, incx);
    ztrsv_(&uplo_char, &trans_char, &diag_char, &n, a, &lda, x, &incx);
    conj(n, x, incx, x, incx);
  }
}

}  // namespace cxxblas
}  // namespace linalg
}  // namespace thunder
