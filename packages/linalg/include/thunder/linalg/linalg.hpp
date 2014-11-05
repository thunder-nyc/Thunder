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
  
  // Const result BLAS level-1 routines
  const T& asum(const T &x, const T &r);
  const T& axpy(const value_type &a, const T &x, const T &y, const T &r);
  const T& copy(const T &x, const T &y);
  const T& dot(const T &x, const T &y, const T &r);
  const T& sdot(const T &x, const T &y, const T &r);
  const T& dotc(const T &x, const T &y, const T &r);
  const T& nrm2(const T &x, const T &r);
  const T& rot(const T &x, const T &y, const value_type &c,
               const value_type &s);
  const T& rotm(const T &x, const T &y, const T &P);
  const T& scal(const value_type &a, const T &x);
  const T& swap(const T &x, const T &y);
  const SizeTensor& iamax(const T &x, const SizeTensor &r);
  const SizeTensor& iamin(const T &x, const SizeTensor &r);
  const T& cabs1(const T &x);

  // Const result BLAS level-2 routines
  const T& gbmv(size_type kl, size_type ku, const value_type &alpha, const T &a,
                const T &x, const value_type &beta, const T&y);
  const T& gemv(const value_type &alpha, const T &a, const T &x,
                const value_type &beta, const T &y);
  const T& ger(const value_type &alpha, const T &x, const T &y, const T &a);
  const T& gerc(const value_type &alpha, const T &x, const T &y, const T &a);
  const T& hbmv(BlasUplo uplo, size_type k, const value_type &alpha,
                const T &a, const T &x, const value_type &beta, const T &y);
  const T& hemv(BlasUplo uplo, const value_type &alpha, const T &a, const T &x,
                const value_type &beta, const T &y);
  const T& her(BlasUplo uplo, const value_type &alpha, const T &x, const T &a);
  const T& her2(BlasUplo uplo, const value_type &alpha, const T &x, const T &y,
               const T &a);
  const T& hpmv(BlasUplo uplo, const value_type &alpha, const T &ap, const T &x,
                const value_type &beta, const T &y);
  const T& hpr(BlasUplo uplo, const value_type &alpha, const T &x, const T &ap);
  const T& hpr2(BlasUplo uplo, const value_type &alpha, const T &x, const T &y,
                const T &ap);
  const T& sbmv(BlasUplo uplo, size_type k, const value_type &alpha,
                const T &a, const T &x, const value_type &beta, const T &y);
  const T& spmv(BlasUplo uplo, const value_type &alpha, const T &ap, const T &x,
                const value_type &beta, const T &y);
  const T& spr(BlasUplo uplo, const value_type &alpha, const T &x, const T &ap);
  const T& spr2(BlasUplo uplo, const value_type &alpha, const T &x, const T &y,
                const T &ap);
  const T& symv(BlasUplo uplo, const value_type &alpha, const T &ap, const T &x,
                const value_type &beta, const T &y);
  const T& syr(BlasUplo uplo, const value_type &alpha, const T &x, const T &a);
  const T& syr2(BlasUplo uplo, const value_type &alpha, const T &x, const T &y,
                const T &a);
  const T& tbmv(BlasUplo uplo, BlasDiag diag, size_type k, const T &a,
                const T &x);
  const T& tbsv(BlasUplo uplo, BlasDiag diag, size_type k, const T &a,
                const T &x);
  const T& tpmv(BlasUplo uplo, BlasDiag diag, const T &ap, const T &x);
  const T& tpsv(BlasUplo uplo, BlasDiag diag, const T &ap, const T &x);
  const T& trmv(BlasUplo uplo, BlasDiag diag, const T &a, const T &x);
  const T& trsv(BlasUplo uplo, BlasDiag diag, const T &a, const T &x);

  // Const result level-3 BLAS routines
  const T& gemm(const value_type &alpha, const T &a, const T&b,
                const value_type &beta, const T &c);
  const T& hemm(BlasUplo uplo, const value_type &alpha, const T &a, const T&b,
                const value_type &beta, const T &c);
  const T& herk(BlasUplo uplo, const value_type &alpha, const T &a,
                const value_type &beta, const T &c);
  const T& herk2(BlasUplo uplo, const value_type &alpha, const T &a, const T&b,
                 const value_type &beta, const T &c);
  const T& symm(BlasUplo uplo, const value_type &alpha, const T &a, const T&b,
                const value_type &beta, const T &c);
  const T& syrk(BlasUplo uplo, const value_type &alpha, const T &a,
                const value_type &beta, const T &c);
  const T& syrk2(BlasUplo uplo, const value_type &alpha, const T &a, const T&b,
                 const value_type &beta, const T &c);
  const T& trmm(BlasUplo uplo, BlasDiag diag, const value_type &alpha,
                const T &a, const T&b, const value_type &beta, const T &c);
  const T& trsm(BlasSide side, BlasUplo uplo, BlasDiag diag,
                const value_type &alpha, const T &a, const T &b);
};

}  // namespace linalg
}  // namespace thunder

#endif  // THUNDER_LINALG_LINALG_HPP_
