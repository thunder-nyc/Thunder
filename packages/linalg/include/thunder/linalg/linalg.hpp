/*
 * \copyright Copyright 2014-2015 Xiang Zhang All Rights Reserved.
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

#include "thunder/linalg/cxxblas.hpp"
#include "thunder/linalg/tensor_type.hpp"
#include "thunder/storage.hpp"
#include "thunder/tensor.hpp"

namespace thunder {
namespace linalg {

template < typename T = DoubleTensor,
           typename H = typename TensorType< T >::handle_type >
class Linalg {
 public:
  typedef T tensor_type;
  typedef H handle_type;

  typedef typename T::allocator_type allocator_type;
  typedef typename T::value_type value_type;
  typedef typename T::size_type size_type;
  typedef typename T::size_storage size_storage;
  typedef SizeTensor size_tensor;

  typedef cxxblas::Uplo Uplo;
  typedef cxxblas::Diag Diag;
  typedef cxxblas::Side Side;

  Linalg();
  template < typename... G >
  explicit Linalg(G... g);
  ~Linalg();

  // Accessors
  H handle();
  void handle(H h);

  // Const linear algebra constructors
  const T& diag(const T &x, const T &r);
  const T& eye(const size_storage &s, const T &r);
  const T& linspace(const value_type &a, const value_type &b, const T &r);
  const T& logspace(const value_type &a, const value_type &b, const T &r);
  const T& tril(const T &x, const T &r);
  const T& triu(const T &x, const T &r);

  // Non-Const linear algebra constructors
  T& diag(const T &x, T &r);
  T& eye(const size_storage &s, T &r);
  T& linspace(const value_type &a, const value_type &b, T &r);
  T& logspace(const value_type &a, const value_type &b, T &r);
  T& tril(const T &x, T &r);
  T& triu(const T &x, T &r);

  // Pointer linear algebra constructors
  T* diag(const T &x, T *r);
  T* eye(const size_storage &s, T *r);
  T* linspace(const value_type &a, const value_type &b, size_type n, T *r);
  T* logspace(const value_type &a, const value_type &b, size_type n, T *r);
  T* tril(const T &x, T *r);
  T* triu(const T &x, T *r);

  // Constructive linear algebra constructors
  T diag(const T &x);
  T eye(const size_storage &s, allocator_type alloc = allocator_type());
  T linspace(const value_type &a, const value_type &b, size_type n = 100,
             allocator_type alloc = allocator_type());
  T logspace(const value_type &a, const value_type &b, size_type n = 100,
             allocator_type alloc = allocator_type());
  T tril(const T &x);
  T triu(const T &x);

  // Const result BLAS level-1 routines
  const T& asum(const T &x, const T &r);
  const T& axpy(const T &x, const T &y, const value_type &a = 1.0);
  const T& copy(const T &x, const T &r);
  const T& dot(const T &x, const T &y, const T &r);
  const T& dotc(const T &x, const T &y, const T &r);
  const T& nrm2(const T &x, const T &r);
  const T& rot(const T &x, const T &y, const value_type &c = 1.0,
               const value_type &s = 1.0);
  const T& rotm(const T &x, const T &y, const T &p);
  const T& scal(const T &x, const value_type &a = 1.0);
  const T& swap(const T &x, const T &y);
  const size_tensor& iamax(const T &x, const size_tensor &r);

  // Non-const result BLAS level-1 routines
  T& asum(const T &x, T &r);
  T& axpy(const T &x, T &y, const value_type &a = 1.0);
  T& copy(const T &x, T &r);
  T& dot(const T &x, const T &y, T &r);
  T& dotc(const T &x, const T &y, T &r);
  T& nrm2(const T &x, T &r);
  T& rot(const T &x, T &y, const value_type &c = 1.0,
         const value_type &s = 1.0);
  T& rotm(const T &x, T &y, const T &p);
  T& scal(T &x, const value_type &a = 1.0);
  T& swap(const T &x, T &y);
  size_tensor& iamax(const T &x, size_tensor &r);

  // Pointer result BLAS level-1 routines
  T* asum(const T &x, T *r);
  T* axpy(const T &x, T *y, const value_type &a = 1.0);
  T* copy(const T &x, T *r);
  T* dot(const T &x, const T &y, T *r);
  T* dotc(const T &x, const T &y, T *r);
  T* nrm2(const T &x, T *r);
  T* rot(const T &x, T *y, const value_type &c = 1.0,
         const value_type &s = 1.0);
  T* rotm(const T &x, T *y, const T &p);
  T* scal(T *x, const value_type &a = 1.0);
  T* swap(const T &x, T *y);
  size_tensor* iamax(const T &x, size_tensor *r);

  // Constructive result BLAS level-1 routines
  T asum(const T &x);
  T axpy(const T &x, const value_type &a = 1.0);
  T copy(const T &x);
  T dot(const T &x, const T &y);
  T dotc(const T &x, const T &y);
  T nrm2(const T &x);
  T rot(const T &x, const value_type &c = 1.0, const value_type &s = 1.0);
  T rotm(const T &x, const T &p);
  T scal(const value_type &a = 1.0, allocator_type alloc = allocator_type());
  T swap(const T &x);
  size_tensor iamax(const T &x);

  // Const result BLAS level-2 routines
  const T& gbmv(const T &a, const T &x, const T &y, size_type kl = 0,
                size_type ku = 0, const value_type &alpha = 1.0,
                const value_type &beta = 0.0);
  const T& gemv(const T &a, const T &x, const T &y,
                const value_type &alpha = 1.0, const value_type &beta = 0.0);
  const T& ger(const T &x, const T &y, const T &a,
               const value_type &alpha = 1.0);
  const T& gerc(const T &x, const T &y, const T &a,
                const value_type &alpha = 1.0);
  const T& hbmv(const T &a, const T &x, const T &y,
                const value_type &alpha = 1.0, size_type k = 0,
                const value_type &beta = 0.0, Uplo uplo = Uplo::kUpper);
  const T& hemv(const T &a, const T &x, const T &y,
                const value_type &alpha = 1.0, const value_type &beta = 0.0,
                Uplo uplo = Uplo::kUpper);
  const T& her(const T &x, const T &a, const value_type &alpha = 1.0,
               Uplo uplo = Uplo::kUpper);
  const T& her2(const T &x, const T &y, const T &a,
                const value_type &alpha = 1.0, Uplo uplo = Uplo::kUpper);
  const T& hpmv(const T &ap, const T &x, const T &y,
                const value_type &alpha = 1.0, const value_type &beta = 0.0,
                Uplo uplo = Uplo::kUpper);
  const T& hpr(const T &x, const T &ap, const value_type &alpha = 1.0,
               Uplo uplo = Uplo::kUpper);
  const T& hpr2(const T &x, const T &y, const T &ap,
                const value_type &alpha = 1.0, Uplo uplo = Uplo::kUpper);
  const T& sbmv(const T &a, const T &x, const T &y,
                size_type k = 0, const value_type &alpha = 1.0,
                const value_type &beta = 0.0, Uplo uplo = Uplo::kUpper);
  const T& spmv(const T &ap, const T &x, const T &y,
                const value_type &alpha = 1.0, const value_type &beta = 0.0,
                Uplo uplo = Uplo::kUpper);
  const T& spr(const T &x, const T &ap, const value_type &alpha = 1.0,
               Uplo uplo = Uplo::kUpper);
  const T& spr2(const T &x, const T &y, const T &ap,
                const value_type &alpha = 1.0, Uplo uplo = Uplo::kUpper);
  const T& symv(const T &ap, const T &x, const T &y,
                const value_type &alpha = 1.0, const value_type &beta = 0.0,
                Uplo uplo = Uplo::kUpper);
  const T& syr(const T &x, const T &a, const value_type &alpha = 1.0,
               Uplo uplo = Uplo::kUpper);
  const T& syr2(const T &x, const T &y, const T &a,
                const value_type &alpha = 1.0, Uplo uplo = Uplo::kUpper);
  const T& tbmv(const T &a, const T &x, size_type k = 0,
                Uplo uplo = Uplo::kUpper, Diag diag = Diag::kNonUnit);
  const T& tbsv(const T &a, const T &x, size_type k = 0,
                Uplo uplo = Uplo::kUpper, Diag diag = Diag::kNonUnit);
  const T& tpmv(const T &ap, const T &x, Uplo uplo = Uplo::kUpper,
                Diag diag = Diag::kNonUnit);
  const T& tpsv(const T &ap, const T &x, Uplo uplo = Uplo::kUpper,
                Diag diag = Diag::kNonUnit);
  const T& trmv(const T &a, const T &x, Uplo uplo = Uplo::kUpper,
                Diag diag = Diag::kNonUnit);
  const T& trsv(const T &a, const T &x, Uplo uplo = Uplo::kUpper,
                Diag diag = Diag::kNonUnit);

  // Const result level-3 BLAS routines
  const T& gemm(const T &a, const T &b, const T &c,
                const value_type &alpha = 1.0, const value_type &beta = 0.0);
  const T& hemm(const T &a, const T &b, const T &c,
                const value_type &alpha = 1.0, const value_type &beta = 0.0,
                Uplo uplo = Uplo::kUpper);
  const T& herk(const T &a, const T &c, const value_type &alpha = 1.0,
                const value_type &beta = 0.0, Uplo uplo = Uplo::kUpper);
  const T& herk2(const T &a, const T &b, const T &c,
                 const value_type &alpha = 1.0, const value_type &beta = 0.0,
                 Uplo uplo = Uplo::kUpper);
  const T& symm(const T &a, const T &b, const T &c,
                const value_type &alpha = 1.0, const value_type &beta = 0.0,
                Uplo uplo = Uplo::kUpper);
  const T& syrk(const T &a, const T &c, const value_type &alpha = 1.0,
                const value_type &beta = 0.0, Uplo uplo = Uplo::kUpper);
  const T& syrk2(const T &a, const T&b, const T &c,
                 const value_type &alpha = 1.0, const value_type &beta = 0.0,
                 Uplo uplo = Uplo::kUpper);
  const T& trmm(const T &a, const T&b, const T &c,
                const value_type &alpha = 1.0, const value_type &beta = 0.0,
                Uplo uplo = Uplo::kUpper, Diag diag = Diag::kNonUnit);
  const T& trsm(const T &a, const T &b, const value_type &alpha = 1.0,
                Side side = Side::kLeft, Uplo uplo = Uplo::kUpper,
                Diag diag = Diag::kNonUnit);

 private:
  H handle_;
};

}  // namespace linalg
}  // namespace thunder

#endif  // THUNDER_LINALG_LINALG_HPP_
