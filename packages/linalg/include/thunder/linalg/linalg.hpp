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

#ifndef THUNDER_LINALG_LINALG_HPP_
#define THUNDER_LINALG_LINALG_HPP_

#include "thunder/tensor.hpp"

namespace thunder {
namespace linalg {

template < typename T = DoubleTensor, typename H = int >
class Linalg {
 public:
  typedef T tensor_type;
  typedef H handle_type;

  typedef typename T::value_type value_type;
  typedef typename T::size_type size_type;
  
  Linalg();
  template < typename... G >
  Linalg(G... g);
  ~Linalg();

  // Needed constant types
  enum BlasUplo {
    BLAS_UPPER,
    BLAS_LOWER
  };
  enum BlasDiag {
    BLAS_NON_UNIT,
    BLAS_UNIT
  };
  enum BlasSide {
    BLAS_RIGHT,
    BLAS_LEFT
  };

  // Const linear algebra constructors
  const T& diag(const T &x, const T &r);
  const T& eye(const T &x, const T &r);
  const T& linspace(const value_type &a, const value_type &b, const T &r,
                    size_type n = 100);
  const T& logspace(const value_type &a, const value_type &b, const T &r,
                    size_type n = 100);
  const T& tril(const T &x, const T &r);
  const T& triu(const T &x, const T &r);

  // Non-const linear algebra constructors
  T& diag(const T &x, T *r);
  T& eye(const T &x, T *r);
  T& linspace(const value_type &a, const value_type &b, T *r,
              size_type n = 100);
  T& logspace(const value_type &a, const value_type &b, T *r,
              size_type n = 100);
  T& tril(const T &x, T *r);
  T& triu(const T &x, T *r);

  // Constructive linear algebra constructors
  T diag(const T &x);
  T eye(const T &x);
  T linspace(const value_type &a, const value_type &b, size_type n = 100);
  T logspace(const value_type &a, const value_type &b, size_type n = 100);
  T tril(const T &x);
  T triu(const T &x);

  // Const result BLAS level-1 routines
  const T& asum(const T &x, const T &r);
  const T& axpy(const T &x, const T &y, const T &r, const value_type &a = 1.0);
  const T& copy(const T &x, const T &r);
  const T& dot(const T &x, const T &y, const T &r);
  const T& sdot(const T &x, const T &y, const T &r);
  const T& dotc(const T &x, const T &y, const T &r);
  const T& nrm2(const T &x, const T &r);
  const T& rot(const T &x, const T &y, const value_type &c = 1.0,
               const value_type &s = 1.0);
  const T& rotm(const T &x, const T &y, const T &P, const T &r);
  const T& scal(const T &x, const T &r, const value_type &a = 1.0);
  const T& swap(const T &x, const T &y);
  const SizeTensor& iamax(const T &x, const SizeTensor &r);
  const SizeTensor& iamin(const T &x, const SizeTensor &r);
  const T& cabs1(const T &x, const T &r);

  // Const result BLAS level-2 routines
  const T& gbmv(const T &a, const T &x, const T &y, const T &r,
                size_type kl = 1, size_type ku = 1,
                const value_type &alpha = 1.0, const value_type &beta = 0.0);
  const T& gemv(const T &a, const T &x, const T &y, const T &r,
                const value_type &alpha = 1.0, const value_type &beta = 0.0);
  const T& ger(const T &x, const T &y, const T &a, const T &r,
               const value_type &alpha = 1.0);
  const T& gerc(const T &x, const T &y, const T &a, const T &r,
                const value_type &alpha = 1.0);
  const T& hbmv(const T &a, const T &x, const T &y, const T &r,
                const value_type &alpha = 1.0, size_type k = 1,
                const value_type &beta = 0.0, BlasUplo uplo = BLAS_UPPER);
  const T& hemv(const T &a, const T &x, const T &y, const T &r,
                const value_type &alpha = 1.0, const value_type &beta = 0.0,
                BlasUplo uplo = BLAS_UPPER);
  const T& her(const T &x, const T &a, const T &r,
               const value_type &alpha = 1.0, BlasUplo uplo = BLAS_UPPER);
  const T& her2(const T &x, const T &y, const T &a, const T &r,
                const value_type &alpha = 1.0, BlasUplo uplo = BLAS_UPPER);
  const T& hpmv(const T &ap, const T &x, const T &y, const T &r,
                const value_type &alpha = 1.0, const value_type &beta = 0.0,
                BlasUplo uplo = BLAS_UPPER);
  const T& hpr(const T &x, const T &ap, const T &r,
               const value_type &alpha = 1.0, BlasUplo uplo = BLAS_UPPER);
  const T& hpr2(const T &x, const T &y, const T &ap, const T &r,
                const value_type &alpha = 1.0, BlasUplo uplo = BLAS_UPPER);
  const T& sbmv(const T &a, const T &x, const T &y, const T &r,
                size_type k = 1, const value_type &alpha = 1.0,
                const value_type &beta = 0.0, BlasUplo uplo = BLAS_UPPER);
  const T& spmv(const T &ap, const T &x, const T &y, const T &r,
                const value_type &alpha = 1.0, const value_type &beta = 0.0,
                BlasUplo uplo = BLAS_UPPER);
  const T& spr(const T &x, const T &ap, const T &r,
               const value_type &alpha = 1.0, BlasUplo uplo = BLAS_UPPER);
  const T& spr2(const T &x, const T &y, const T &ap, const T &r,
                const value_type &alpha = 1.0, BlasUplo uplo = BLAS_UPPER);
  const T& symv(const T &ap, const T &x, const T &y, const T &r,
                const value_type &alpha = 1.0, const value_type &beta = 0.0,
                BlasUplo uplo = BLAS_UPPER);
  const T& syr(const T &x, const T &a, const T &r,
               const value_type &alpha = 1.0, BlasUplo uplo = BLAS_UPPER);
  const T& syr2(const T &x, const T &y, const T &a, const T &r,
                const value_type &alpha = 1.0, BlasUplo uplo = BLAS_UPPER);
  const T& tbmv(const T &a, const T &x, const T &r, size_type k = 1,
                BlasUplo uplo = BLAS_UPPER, BlasDiag diag = BLAS_NON_UNIT);
  const T& tbsv(const T &a, const T &x, const T &r, size_type k = 1,
                BlasUplo uplo = BLAS_UPPER, BlasDiag diag = BLAS_NON_UNIT);
  const T& tpmv(const T &ap, const T &x, const T &r,
                BlasUplo uplo = BLAS_UPPER, BlasDiag diag = BLAS_NON_UNIT);
  const T& tpsv(const T &ap, const T &x, const T &r,
                BlasUplo uplo = BLAS_UPPER, BlasDiag diag = BLAS_NON_UNIT);
  const T& trmv(const T &a, const T &x, const T &r,
                BlasUplo uplo = BLAS_UPPER, BlasDiag diag = BLAS_NON_UNIT);
  const T& trsv(const T &a, const T &x, const T &r,
                BlasUplo uplo = BLAS_UPPER, BlasDiag diag = BLAS_NON_UNIT);

  // Const result level-3 BLAS routines
  const T& gemm(const T &a, const T&b, const T &c, const T &r,
                const value_type &alpha = 1.0, const value_type &beta = 0.0);
  const T& hemm(const T &a, const T&b, const T &c, const T &r,
                const value_type &alpha = 1.0, const value_type &beta = 0.0
                BlasUplo uplo = BLAS_UPPER);
  const T& herk(const T &a, const T &c, const T &r,
                const value_type &alpha = 1.0, const value_type &beta = 0.0,
                BlasUplo uplo = BLAS_UPPER);
  const T& herk2(const T &a, const T&b, const T &c, const T &r,
                 const value_type &alpha = 1.0, const value_type &beta = 0.0,
                 BlasUplo uplo = BLAS_UPPER);
  const T& symm(const T &a, const T&b, const T &c, const T &r,
                const value_type &alpha = 1.0, const value_type &beta = 0.0
                BlasUplo uplo = BLAS_UPPER);
  const T& syrk(const T &a, const T &c, const T &r,
                const value_type &alpha = 1.0, const value_type &beta = 0.0,
                BlasUplo uplo = BLAS_UPPER);
  const T& syrk2(const T &a, const T&b, const T &c, const T &r,
                 const value_type &alpha = 1.0, const value_type &beta = 0.0,
                 BlasUplo uplo = BLAS_UPPER);
  const T& trmm(const T &a, const T&b, const T &c, const T &r,
                const value_type &alpha = 1.0, const value_type &beta = 0.0
                BlasUplo uplo = BLAS_UPPER, BlasDiag diag = BLAS_NON_UNIT);
  const T& trsm(const T &a, const T &b, const T &r,
                const value_type &alpha = 1.0, BlasSide side = BLAS_LEFT,
                BlasUplo uplo = BLAS_UPPER, BlasDiag diag = BLAS_NON_UNIT);

 private:
  H handle_;
};

}  // namespace linalg
}  // namespace thunder

#endif  // THUNDER_LINALG_LINALG_HPP_
