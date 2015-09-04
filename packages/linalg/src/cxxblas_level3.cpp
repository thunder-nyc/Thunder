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

#include "thunder/linalg/blas.hpp"

namespace thunder {
namespace linalg {
namespace cxxblas {

static inline char heTransChar(Order order, Trans trans) {
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
      return 'C';
    } else if (trans == Trans::kTrans) {
      return 'N';
    } else if (trans == Trans::kConjTrans) {
      return 'N';
    }
  }
  return '\0';
}

static inline char syTransChar(Order order, Trans trans) {
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

static inline char sideChar(Order order, Side side) {
  if (order == Order::kColMajor) {
    if (side == Side::kLeft) {
      return 'L';
    } else if (side == Side::kRight) {
      return 'R';
    }
  } else if (order == Order::kRowMajor) {
    if (side == Side::kLeft) {
      return 'R';
    } else if (side == Side::kRight) {
      return 'L';
    }
  }
  return '\0';
}

static inline char diagChar(Diag diag) {
  if (diag == Diag::kNonUnit) {
    return 'N';
  } else if (diag == Diag::kUnit) {
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
  if (order == Order::kColMajor) {
    if (lda == 0) {
      if (transa == Trans::kNoTrans) {
        lda = ::std::max(1, m);
      } else {
        lda = ::std::max(1, k);
      }
    }
    if (ldb == 0) {
      if (transb == Trans::kNoTrans) {
        ldb = ::std::max(1, k);
      } else {
        ldb = ::std::max(1, n);
      }
    }
    if (ldc == 0) {
      ldc = ::std::max(1, m);
    }
    sgemm_(&transa_char, &transb_char, &m, &n, &k, &alpha, a, &lda, b, &ldb,
           &beta, c, &ldc);
  } else {
    if (lda == 0) {
      if (transa == Trans::kNoTrans) {
        lda = ::std::max(1, k);
      } else {
        lda = ::std::max(1, m);
      }
    }
    if (ldb == 0) {
      if (transb == Trans::kNoTrans) {
        ldb = ::std::max(1, n);
      } else {
        ldb = ::std::max(1, k);
      }
    }
    if (ldc == 0) {
      ldc = ::std::max(1, n);
    }
    sgemm_(&transb_char, &transa_char, &n, &m, &k, &alpha, b, &ldb, a, &lda,
           &beta, c, &ldc);
  }
}

void gemm(int m, int n, int k, const double *a, const double *b, double *c,
          double alpha, double beta, int lda, int ldb, int ldc, Order order,
          Trans transa, Trans transb) {
  char transa_char = transChar(transa);
  char transb_char = transChar(transb);
  if (order == Order::kColMajor) {
    if (lda == 0) {
      if (transa == Trans::kNoTrans) {
        lda = ::std::max(1, m);
      } else {
        lda = ::std::max(1, k);
      }
    }
    if (ldb == 0) {
      if (transb == Trans::kNoTrans) {
        ldb = ::std::max(1, k);
      } else {
        ldb = ::std::max(1, n);
      }
    }
    if (ldc == 0) {
      ldc = ::std::max(1, m);
    }
    dgemm_(&transa_char, &transb_char, &m, &n, &k, &alpha, a, &lda, b, &ldb,
           &beta, c, &ldc);
  } else {
    if (lda == 0) {
      if (transa == Trans::kNoTrans) {
        lda = ::std::max(1, k);
      } else {
        lda = ::std::max(1, m);
      }
    }
    if (ldb == 0) {
      if (transb == Trans::kNoTrans) {
        ldb = ::std::max(1, n);
      } else {
        ldb = ::std::max(1, k);
      }
    }
    if (ldc == 0) {
      ldc = ::std::max(1, n);
    }
    dgemm_(&transb_char, &transa_char, &n, &m, &k, &alpha, b, &ldb, a, &lda,
           &beta, c, &ldc);
  }
}

void gemm(int m, int n, int k, const ::std::complex< float > *a,
          const ::std::complex< float > *b, ::std::complex< float > *c,
          ::std::complex< float > alpha, ::std::complex< float > beta, int lda,
          int ldb, int ldc, Order order, Trans transa, Trans transb) {
  char transa_char = transChar(transa);
  char transb_char = transChar(transb);
  if (order == Order::kColMajor) {
    if (lda == 0) {
      if (transa == Trans::kNoTrans) {
        lda = ::std::max(1, m);
      } else {
        lda = ::std::max(1, k);
      }
    }
    if (ldb == 0) {
      if (transb == Trans::kNoTrans) {
        ldb = ::std::max(1, k);
      } else {
        ldb = ::std::max(1, n);
      }
    }
    if (ldc == 0) {
      ldc = ::std::max(1, m);
    }
    cgemm_(&transa_char, &transb_char, &m, &n, &k, &alpha, a, &lda, b, &ldb,
           &beta, c, &ldc);
  } else {
    if (lda == 0) {
      if (transa == Trans::kNoTrans) {
        lda = ::std::max(1, k);
      } else {
        lda = ::std::max(1, m);
      }
    }
    if (ldb == 0) {
      if (transb == Trans::kNoTrans) {
        ldb = ::std::max(1, n);
      } else {
        ldb = ::std::max(1, k);
      }
    }
    if (ldc == 0) {
      ldc = ::std::max(1, n);
    }
    cgemm_(&transb_char, &transa_char, &n, &m, &k, &alpha, b, &ldb, a, &lda,
           &beta, c, &ldc);
  }
}

void gemm(int m, int n, int k, const ::std::complex< double > *a,
          const ::std::complex< double > *b, ::std::complex< double > *c,
          ::std::complex< double > alpha, ::std::complex< double > beta,
          int lda, int ldb, int ldc, Order order, Trans transa, Trans transb) {
  char transa_char = transChar(transa);
  char transb_char = transChar(transb);
  if (order == Order::kColMajor) {
    if (lda == 0) {
      if (transa == Trans::kNoTrans) {
        lda = ::std::max(1, m);
      } else {
        lda = ::std::max(1, k);
      }
    }
    if (ldb == 0) {
      if (transb == Trans::kNoTrans) {
        ldb = ::std::max(1, k);
      } else {
        ldb = ::std::max(1, n);
      }
    }
    if (ldc == 0) {
      ldc = ::std::max(1, m);
    }
    zgemm_(&transa_char, &transb_char, &m, &n, &k, &alpha, a, &lda, b, &ldb,
           &beta, c, &ldc);
  } else {
    if (lda == 0) {
      if (transa == Trans::kNoTrans) {
        lda = ::std::max(1, k);
      } else {
        lda = ::std::max(1, m);
      }
    }
    if (ldb == 0) {
      if (transb == Trans::kNoTrans) {
        ldb = ::std::max(1, n);
      } else {
        ldb = ::std::max(1, k);
      }
    }
    if (ldc == 0) {
      ldc = ::std::max(1, n);
    }
    zgemm_(&transb_char, &transa_char, &n, &m, &k, &alpha, b, &ldb, a, &lda,
           &beta, c, &ldc);
  }
}

void hemm(int m, int n, const float *a, const float *b, float *c, float alpha,
          float beta, int lda, int ldb, int ldc, Order order, Side side,
          Uplo uplo) {
  symm(m, n, a, b, c, alpha, beta, lda, ldb, ldc, order, side, uplo);
}

void hemm(int m, int n, const double *a, const double *b, double *c,
          double alpha, double beta, int lda, int ldb, int ldc, Order order,
          Side side, Uplo uplo) {
  symm(m, n, a, b, c, alpha, beta, lda, ldb, ldc, order, side, uplo);
}

void hemm(int m, int n, const ::std::complex< float > *a,
          const ::std::complex< float > *b, ::std::complex< float > *c,
          ::std::complex< float > alpha, ::std::complex< float > beta, int lda,
          int ldb, int ldc, Order order, Side side, Uplo uplo) {
  if (lda == 0) {
    if (side == Side::kLeft) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
  char side_char = sideChar(order, side);
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    if (ldb == 0) {
      ldb = ::std::max(1, m);
    }
    if (ldc == 0) {
      ldc = ::std::max(1, m);
    }
    chemm_(&side_char, &uplo_char, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c,
           &ldc);
  } else {
    if (ldb == 0) {
      ldb = ::std::max(1, n);
    }
    if (ldc == 0) {
      ldc = ::std::max(1, n);
    }
    chemm_(&side_char, &uplo_char, &n, &m, &alpha, a, &lda, b, &ldb, &beta, c,
           &ldc);
  }
}

void hemm(int m, int n, const ::std::complex< double > *a,
          const ::std::complex< double > *b, ::std::complex< double > *c,
          ::std::complex< double > alpha, ::std::complex< double > beta,
          int lda, int ldb, int ldc, Order order, Side side, Uplo uplo) {
  if (lda == 0) {
    if (side == Side::kLeft) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
  char side_char = sideChar(order, side);
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    if (ldb == 0) {
      ldb = ::std::max(1, m);
    }
    if (ldc == 0) {
      ldc = ::std::max(1, m);
    }
    zhemm_(&side_char, &uplo_char, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c,
           &ldc);
  } else {
    if (ldb == 0) {
      ldb = ::std::max(1, n);
    }
    if (ldc == 0) {
      ldc = ::std::max(1, n);
    }
    zhemm_(&side_char, &uplo_char, &n, &m, &alpha, a, &lda, b, &ldb, &beta, c,
           &ldc);
  }
}

void herk(int n, int k, const float *a, float *c, float alpha, float beta,
          int lda, int ldc, Order order, Uplo uplo, Trans trans) {
  syrk(n, k, a, c, alpha, beta, lda, ldc, order, uplo, trans);
}

void herk(int n, int k, const double *a, double *c, double alpha, double beta,
          int lda, int ldc, Order order, Uplo uplo, Trans trans) {
  syrk(n, k, a, c, alpha, beta, lda, ldc, order, uplo, trans);
}

void herk(int n, int k, const ::std::complex< float > *a,
          ::std::complex< float > *c, float alpha, float beta, int lda, int ldc,
          Order order, Uplo uplo, Trans trans) {
  if (lda == 0) {
    if (order == Order::kColMajor && trans == Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else if (order == Order::kRowMajor && trans != Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else {
      lda = ::std::max(1, k);
    }
  }
  if (ldc == 0) {
    ldc = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  char trans_char = heTransChar(order, trans);
  cherk_(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

void herk(int n, int k, const ::std::complex< double > *a,
          ::std::complex< double > *c, double alpha, double beta, int lda,
          int ldc, Order order, Uplo uplo, Trans trans) {
  if (lda == 0) {
    if (order == Order::kColMajor && trans == Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else if (order == Order::kRowMajor && trans != Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else {
      lda = ::std::max(1, k);
    }
  }
  if (ldc == 0) {
    ldc = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  char trans_char = heTransChar(order, trans);
  zherk_(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

void her2k(int n, int k, const float *a, const float *b, float *c,
           float alpha, float beta, int lda, int ldb, int ldc, Order order,
           Uplo uplo, Trans trans) {
  syr2k(n, k, a, b, c, alpha, beta, lda, ldb, ldc, order, uplo, trans);
}

void her2k(int n, int k, const double *a, const double *b, double *c,
           double alpha, double beta, int lda, int ldb, int ldc, Order order,
           Uplo uplo, Trans trans) {
  syr2k(n, k, a, b, c, alpha, beta, lda, ldb, ldc, order, uplo, trans);
}

void her2k(int n, int k, const ::std::complex< float > *a,
           const ::std::complex< float > *b, ::std::complex< float > *c,
           ::std::complex< float > alpha, ::std::complex< float > beta, int lda,
           int ldb, int ldc, Order order, Uplo uplo, Trans trans) {
  if (lda == 0) {
    if (order == Order::kColMajor && trans == Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else if (order == Order::kRowMajor && trans != Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else {
      lda = ::std::max(1, k);
    }
  }
  if (ldb == 0) {
    if (order == Order::kColMajor && trans == Trans::kNoTrans) {
      ldb = ::std::max(1, n);
    } else if (order == Order::kRowMajor && trans != Trans::kNoTrans) {
      ldb = ::std::max(1, n);
    } else {
      ldb = ::std::max(1, k);
    }
  }
  if (ldc == 0) {
    ldc = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  char trans_char = heTransChar(order, trans);
  cher2k_(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c,
          &ldc);
}

void her2k(int n, int k, const ::std::complex< double > *a,
           const ::std::complex< double > *b, ::std::complex< double > *c,
           ::std::complex< double > alpha, ::std::complex< double > beta,
           int lda, int ldb, int ldc, Order order, Uplo uplo, Trans trans) {
  if (lda == 0) {
    if (order == Order::kColMajor && trans == Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else if (order == Order::kRowMajor && trans != Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else {
      lda = ::std::max(1, k);
    }
  }
  if (ldb == 0) {
    if (order == Order::kColMajor && trans == Trans::kNoTrans) {
      ldb = ::std::max(1, n);
    } else if (order == Order::kRowMajor && trans != Trans::kNoTrans) {
      ldb = ::std::max(1, n);
    } else {
      ldb = ::std::max(1, k);
    }
  }
  if (ldc == 0) {
    ldc = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  char trans_char = heTransChar(order, trans);
  zher2k_(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c,
          &ldc);
}

void symm(int m, int n, const float *a, const float *b, float *c,
          float alpha, float beta, int lda, int ldb, int ldc, Order order,
          Side side, Uplo uplo) {
  if (lda == 0) {
    if (side == Side::kLeft) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
  char side_char = sideChar(order, side);
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    if (ldb == 0) {
      ldb = ::std::max(1, m);
    }
    if (ldc == 0) {
      ldc = ::std::max(1, m);
    }
    ssymm_(&side_char, &uplo_char, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c,
           &ldc);
  } else {
    if (ldb == 0) {
      ldb = ::std::max(1, n);
    }
    if (ldc == 0) {
      ldc = ::std::max(1, n);
    }
    ssymm_(&side_char, &uplo_char, &n, &m, &alpha, a, &lda, b, &ldb, &beta, c,
           &ldc);
  }
}

void symm(int m, int n, const double *a, const double *b, double *c,
          double alpha, double beta, int lda, int ldb, int ldc,
          Order order, Side side, Uplo uplo) {
  if (lda == 0) {
    if (side == Side::kLeft) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
  char side_char = sideChar(order, side);
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    if (ldb == 0) {
      ldb = ::std::max(1, m);
    }
    if (ldc == 0) {
      ldc = ::std::max(1, m);
    }
    dsymm_(&side_char, &uplo_char, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c,
           &ldc);
  } else {
    if (ldb == 0) {
      ldb = ::std::max(1, n);
    }
    if (ldc == 0) {
      ldc = ::std::max(1, n);
    }
    dsymm_(&side_char, &uplo_char, &n, &m, &alpha, a, &lda, b, &ldb, &beta, c,
           &ldc);
  }
}

void symm(int m, int n, const ::std::complex< float > *a,
          const ::std::complex< float > *b, ::std::complex< float > *c,
          ::std::complex< float > alpha, ::std::complex< float > beta, int lda,
          int ldb, int ldc, Order order, Side side, Uplo uplo) {
  if (lda == 0) {
    if (side == Side::kLeft) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
  char side_char = sideChar(order, side);
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    if (ldb == 0) {
      ldb = ::std::max(1, m);
    }
    if (ldc == 0) {
      ldc = ::std::max(1, m);
    }
    csymm_(&side_char, &uplo_char, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c,
           &ldc);
  } else {
    if (ldb == 0) {
      ldb = ::std::max(1, n);
    }
    if (ldc == 0) {
      ldc = ::std::max(1, n);
    }
    csymm_(&side_char, &uplo_char, &n, &m, &alpha, a, &lda, b, &ldb, &beta, c,
           &ldc);
  }
}

void symm(int m, int n, const ::std::complex< double > *a,
          const ::std::complex< double > *b, ::std::complex< double > *c,
          ::std::complex< double > alpha, ::std::complex< double > beta,
          int lda, int ldb, int ldc, Order order, Side side, Uplo uplo) {
  if (lda == 0) {
    if (side == Side::kLeft) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
  char side_char = sideChar(order, side);
  char uplo_char = uploChar(order, uplo);
  if (order == Order::kColMajor) {
    if (ldb == 0) {
      ldb = ::std::max(1, m);
    }
    if (ldc == 0) {
      ldc = ::std::max(1, m);
    }
    zsymm_(&side_char, &uplo_char, &m, &n, &alpha, a, &lda, b, &ldb, &beta, c,
           &ldc);
  } else {
    if (ldb == 0) {
      ldb = ::std::max(1, n);
    }
    if (ldc == 0) {
      ldc = ::std::max(1, n);
    }
    zsymm_(&side_char, &uplo_char, &n, &m, &alpha, a, &lda, b, &ldb, &beta, c,
           &ldc);
  }
}

void syrk(int n, int k, const float *a, float *c, float alpha, float beta,
          int lda, int ldc, Order order, Uplo uplo, Trans trans) {
  if (lda == 0) {
    if (order == Order::kColMajor && trans == Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else if (order == Order::kRowMajor && trans != Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else {
      lda = ::std::max(1, k);
    }
  }
  if (ldc == 0) {
    ldc = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  char trans_char = syTransChar(order, trans);
  ssyrk_(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

void syrk(int n, int k, const double *a, double *c, double alpha, double beta,
          int lda, int ldc, Order order, Uplo uplo, Trans trans) {
  if (lda == 0) {
    if (order == Order::kColMajor && trans == Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else if (order == Order::kRowMajor && trans != Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else {
      lda = ::std::max(1, k);
    }
  }
  if (ldc == 0) {
    ldc = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  char trans_char = syTransChar(order, trans);
  dsyrk_(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

void syrk(int n, int k, const ::std::complex< float > *a,
          ::std::complex< float > *c, ::std::complex< float > alpha,
          ::std::complex< float > beta, int lda, int ldc, Order order,
          Uplo uplo, Trans trans) {
  if (lda == 0) {
    if (order == Order::kColMajor && trans == Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else if (order == Order::kRowMajor && trans != Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else {
      lda = ::std::max(1, k);
    }
  }
  if (ldc == 0) {
    ldc = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  char trans_char = syTransChar(order, trans);
  csyrk_(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

void syrk(int n, int k, const ::std::complex< double > *a,
          ::std::complex< double > *c, ::std::complex< double > alpha,
          ::std::complex< double > beta, int lda, int ldc, Order order,
          Uplo uplo, Trans trans) {
  if (lda == 0) {
    if (order == Order::kColMajor && trans == Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else if (order == Order::kRowMajor && trans != Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else {
      lda = ::std::max(1, k);
    }
  }
  if (ldc == 0) {
    ldc = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  char trans_char = syTransChar(order, trans);
  zsyrk_(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, &beta, c, &ldc);
}

void syr2k(int n, int k, const float *a, const float *b, float *c,
           float alpha, float beta, int lda, int ldb, int ldc, Order order,
           Uplo uplo, Trans trans) {
  if (lda == 0) {
    if (order == Order::kColMajor && trans == Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else if (order == Order::kRowMajor && trans != Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else {
      lda = ::std::max(1, k);
    }
  }
  if (ldb == 0) {
    if (order == Order::kColMajor && trans == Trans::kNoTrans) {
      ldb = ::std::max(1, n);
    } else if (order == Order::kRowMajor && trans != Trans::kNoTrans) {
      ldb = ::std::max(1, n);
    } else {
      ldb = ::std::max(1, k);
    }
  }
  if (ldc == 0) {
    ldc = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  char trans_char = syTransChar(order, trans);
  ssyr2k_(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c,
          &ldc);
}

void syr2k(int n, int k, const double *a, const double *b, double *c,
           double alpha, double beta, int lda, int ldb, int ldc, Order order,
           Uplo uplo, Trans trans) {
  if (lda == 0) {
    if (order == Order::kColMajor && trans == Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else if (order == Order::kRowMajor && trans != Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else {
      lda = ::std::max(1, k);
    }
  }
  if (ldb == 0) {
    if (order == Order::kColMajor && trans == Trans::kNoTrans) {
      ldb = ::std::max(1, n);
    } else if (order == Order::kRowMajor && trans != Trans::kNoTrans) {
      ldb = ::std::max(1, n);
    } else {
      ldb = ::std::max(1, k);
    }
  }
  if (ldc == 0) {
    ldc = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  char trans_char = syTransChar(order, trans);
  dsyr2k_(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c,
          &ldc);
}

void syr2k(int n, int k, const ::std::complex< float > *a,
           const ::std::complex< float > *b, ::std::complex< float > *c,
           ::std::complex< float > alpha, ::std::complex< float > beta, int lda,
           int ldb, int ldc, Order order, Uplo uplo, Trans trans) {
  if (lda == 0) {
    if (order == Order::kColMajor && trans == Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else if (order == Order::kRowMajor && trans != Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else {
      lda = ::std::max(1, k);
    }
  }
  if (ldb == 0) {
    if (order == Order::kColMajor && trans == Trans::kNoTrans) {
      ldb = ::std::max(1, n);
    } else if (order == Order::kRowMajor && trans != Trans::kNoTrans) {
      ldb = ::std::max(1, n);
    } else {
      ldb = ::std::max(1, k);
    }
  }
  if (ldc == 0) {
    ldc = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  char trans_char = syTransChar(order, trans);
  csyr2k_(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c,
          &ldc);
}

void syr2k(int n, int k, const ::std::complex< double > *a,
           const ::std::complex< double > *b, ::std::complex< double > *c,
           ::std::complex< double > alpha, ::std::complex< double > beta,
           int lda, int ldb, int ldc, Order order, Uplo uplo, Trans trans) {
  if (lda == 0) {
    if (order == Order::kColMajor && trans == Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else if (order == Order::kRowMajor && trans != Trans::kNoTrans) {
      lda = ::std::max(1, n);
    } else {
      lda = ::std::max(1, k);
    }
  }
  if (ldb == 0) {
    if (order == Order::kColMajor && trans == Trans::kNoTrans) {
      ldb = ::std::max(1, n);
    } else if (order == Order::kRowMajor && trans != Trans::kNoTrans) {
      ldb = ::std::max(1, n);
    } else {
      ldb = ::std::max(1, k);
    }
  }
  if (ldc == 0) {
    ldc = ::std::max(1, n);
  }
  char uplo_char = uploChar(order, uplo);
  char trans_char = syTransChar(order, trans);
  zsyr2k_(&uplo_char, &trans_char, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c,
          &ldc);
}

void trmm(int m, int n, const float *a, float *b, float alpha, int lda, int ldb,
          Order order, Side side, Uplo uplo, Trans transa, Diag diag) {
  if (lda == 0) {
    if (side == Side::kLeft) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
  if (ldb == 0) {
    if (order == Order::kColMajor) {
      ldb = ::std::max(1, m);
    } else {
      ldb = ::std::max(1, n);
    }
  }
  char side_char = sideChar(order, side);
  char uplo_char = uploChar(order, uplo);
  char transa_char = transChar(transa);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor) {
    strmm_(&side_char, &uplo_char, &transa_char, &diag_char, &m, &n, &alpha, a,
           &lda, b, &ldb);
  } else {
    strmm_(&side_char, &uplo_char, &transa_char, &diag_char, &n, &m, &alpha, a,
           &lda, b, &ldb);
  }
}

void trmm(int m, int n, const double *a, double *b, double alpha, int lda,
          int ldb, Order order, Side side, Uplo uplo, Trans transa, Diag diag) {
  if (lda == 0) {
    if (side == Side::kLeft) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
  if (ldb == 0) {
    if (order == Order::kColMajor) {
      ldb = ::std::max(1, m);
    } else {
      ldb = ::std::max(1, n);
    }
  }
  char side_char = sideChar(order, side);
  char uplo_char = uploChar(order, uplo);
  char transa_char = transChar(transa);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor) {
    dtrmm_(&side_char, &uplo_char, &transa_char, &diag_char, &m, &n, &alpha, a,
           &lda, b, &ldb);
  } else {
    dtrmm_(&side_char, &uplo_char, &transa_char, &diag_char, &n, &m, &alpha, a,
           &lda, b, &ldb);
  }
}

void trmm(int m, int n, const ::std::complex< float > *a,
          ::std::complex< float > *b, ::std::complex< float > alpha, int lda,
          int ldb, Order order, Side side, Uplo uplo, Trans transa, Diag diag) {
  if (lda == 0) {
    if (side == Side::kLeft) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
  if (ldb == 0) {
    if (order == Order::kColMajor) {
      ldb = ::std::max(1, m);
    } else {
      ldb = ::std::max(1, n);
    }
  }
  char side_char = sideChar(order, side);
  char uplo_char = uploChar(order, uplo);
  char transa_char = transChar(transa);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor) {
    ctrmm_(&side_char, &uplo_char, &transa_char, &diag_char, &m, &n, &alpha, a,
           &lda, b, &ldb);
  } else {
    ctrmm_(&side_char, &uplo_char, &transa_char, &diag_char, &n, &m, &alpha, a,
           &lda, b, &ldb);
  }
}

void trmm(int m, int n, const ::std::complex< double > *a,
          ::std::complex< double > *b, ::std::complex< double > alpha, int lda,
          int ldb, Order order, Side side, Uplo uplo, Trans transa, Diag diag) {
  if (lda == 0) {
    if (side == Side::kLeft) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
  if (ldb == 0) {
    if (order == Order::kColMajor) {
      ldb = ::std::max(1, m);
    } else {
      ldb = ::std::max(1, n);
    }
  }
  char side_char = sideChar(order, side);
  char uplo_char = uploChar(order, uplo);
  char transa_char = transChar(transa);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor) {
    ztrmm_(&side_char, &uplo_char, &transa_char, &diag_char, &m, &n, &alpha, a,
           &lda, b, &ldb);
  } else {
    ztrmm_(&side_char, &uplo_char, &transa_char, &diag_char, &n, &m, &alpha, a,
           &lda, b, &ldb);
  }
}

void trsm(int m, int n, const float *a, float *b, float alpha, int lda, int ldb,
          Order order, Side side, Uplo uplo, Trans transa, Diag diag) {
  if (lda == 0) {
    if (side == Side::kLeft) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
  if (ldb == 0) {
    if (order == Order::kColMajor) {
      ldb = ::std::max(1, m);
    } else {
      ldb = ::std::max(1, n);
    }
  }
  char side_char = sideChar(order, side);
  char uplo_char = uploChar(order, uplo);
  char transa_char = transChar(transa);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor) {
    strsm_(&side_char, &uplo_char, &transa_char, &diag_char, &m, &n, &alpha, a,
           &lda, b, &ldb);
  } else {
    strsm_(&side_char, &uplo_char, &transa_char, &diag_char, &n, &m, &alpha, a,
           &lda, b, &ldb);
  }
}

void trsm(int m, int n, const double *a, double *b, double alpha, int lda,
          int ldb, Order order, Side side, Uplo uplo, Trans transa, Diag diag) {
  if (lda == 0) {
    if (side == Side::kLeft) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
  if (ldb == 0) {
    if (order == Order::kColMajor) {
      ldb = ::std::max(1, m);
    } else {
      ldb = ::std::max(1, n);
    }
  }
  char side_char = sideChar(order, side);
  char uplo_char = uploChar(order, uplo);
  char transa_char = transChar(transa);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor) {
    dtrsm_(&side_char, &uplo_char, &transa_char, &diag_char, &m, &n, &alpha, a,
           &lda, b, &ldb);
  } else {
    dtrsm_(&side_char, &uplo_char, &transa_char, &diag_char, &n, &m, &alpha, a,
           &lda, b, &ldb);
  }
}

void trsm(int m, int n, const ::std::complex< float > *a,
          ::std::complex< float > *b, ::std::complex< float > alpha, int lda,
          int ldb, Order order, Side side, Uplo uplo, Trans transa, Diag diag) {
  if (lda == 0) {
    if (side == Side::kLeft) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
  if (ldb == 0) {
    if (order == Order::kColMajor) {
      ldb = ::std::max(1, m);
    } else {
      ldb = ::std::max(1, n);
    }
  }
  char side_char = sideChar(order, side);
  char uplo_char = uploChar(order, uplo);
  char transa_char = transChar(transa);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor) {
    ctrsm_(&side_char, &uplo_char, &transa_char, &diag_char, &m, &n, &alpha, a,
           &lda, b, &ldb);
  } else {
    ctrsm_(&side_char, &uplo_char, &transa_char, &diag_char, &n, &m, &alpha, a,
           &lda, b, &ldb);
  }
}

void trsm(int m, int n, const ::std::complex< double > *a,
          ::std::complex< double > *b, ::std::complex< double > alpha, int lda,
          int ldb, Order order, Side side, Uplo uplo, Trans transa, Diag diag) {
  if (lda == 0) {
    if (side == Side::kLeft) {
      lda = ::std::max(1, m);
    } else {
      lda = ::std::max(1, n);
    }
  }
  if (ldb == 0) {
    if (order == Order::kColMajor) {
      ldb = ::std::max(1, m);
    } else {
      ldb = ::std::max(1, n);
    }
  }
  char side_char = sideChar(order, side);
  char uplo_char = uploChar(order, uplo);
  char transa_char = transChar(transa);
  char diag_char = diagChar(diag);
  if (order == Order::kColMajor) {
    ztrsm_(&side_char, &uplo_char, &transa_char, &diag_char, &m, &n, &alpha, a,
           &lda, b, &ldb);
  } else {
    ztrsm_(&side_char, &uplo_char, &transa_char, &diag_char, &n, &m, &alpha, a,
           &lda, b, &ldb);
  }
}

}  // namespace cxxblas
}  // namespace linalg
}  // namespace thunder
