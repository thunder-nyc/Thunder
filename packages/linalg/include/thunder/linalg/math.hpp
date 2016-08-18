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

#ifndef THUNDER_LINALG_MATH_HPP_
#define THUNDER_LINALG_MATH_HPP_

#include "thunder/tensor.hpp"

namespace thunder {
namespace linalg {
namespace math {

// Linear algebra constructors
template < typename L >
const typename L::tensor_type& diag(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &r);
template < typename L >
const typename L::tensor_type& eye(
    L *l, const typename L::size_storage &s, const typename L::tensor_type &r);
template < typename L >
const typename L::tensor_type& linspace(
    L *l, const typename L::value_type &a, const typename L::value_type &b,
    const typename L::tensor_type &r);
template < typename L >
const typename L::tensor_type& logspace(
    L *l, const typename L::value_type &a, const typename L::value_type &b,
    const typename L::tensor_type &r);
template < typename L >
const typename L::tensor_type& tril(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &r);
template < typename L >
const typename L::tensor_type& triu(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &r);
template < typename L >
typename L::tensor_type* diag(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *r);
template < typename L >
typename L::tensor_type* eye(
    L *l, const typename L::size_storage &s, typename L::tensor_type *r);
template < typename L >
typename L::tensor_type* linspace(
    L *l, const typename L::value_type &a, const typename L::value_type &b,
    typename L::size_type n, typename L::tensor_type *r);
template < typename L >
typename L::tensor_type* logspace(
    L *l, const typename L::value_type &a, const typename L::value_type &b,
    typename L::size_type n, typename L::tensor_type *r);
template < typename L >
typename L::tensor_type* tril(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *r);
template < typename L >
typename L::tensor_type* triu(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *r);

// BLAS level-1 routines
template < typename L >
const typename L::tensor_type& asum(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &r);
template < typename L >
const typename L::tensor_type& axpy(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::value_type &a);
template < typename L >
const typename L::tensor_type& copy(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &r);
template < typename L >
const typename L::tensor_type& dot(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &r);
template < typename L >
const typename L::tensor_type& dotc(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &r);
template < typename L >
const typename L::tensor_type& nrm2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &r);
template < typename L >
const typename L::tensor_type& rot(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::value_type &c, const typename L::value_type &s);
template < typename L >
const typename L::tensor_type& rotm(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &p);
template < typename L >
const typename L::tensor_type& scal(
    L *l, const typename L::tensor_type &x, const typename L::value_type &a);
template < typename L >
const typename L::tensor_type& swap(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y);
template < typename L >
const typename L::size_tensor& iamax(
    L *l, const typename L::tensor_type &x, const typename L::size_tensor &r);
template < typename L >
typename L::tensor_type* asum(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *r);
template < typename L >
typename L::tensor_type* axpy(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *y,
    const typename L::value_type &a);
template < typename L >
typename L::tensor_type* copy(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *r);
template < typename L >
typename L::tensor_type* dot(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    typename L::tensor_type *r);
template < typename L >
typename L::tensor_type* dotc(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    typename L::tensor_type *r);
template < typename L >
typename L::tensor_type* nrm2(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *r);
template < typename L >
typename L::tensor_type* rot(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *y,
    const typename L::value_type &c, const typename L::value_type &s);
template < typename L >
typename L::tensor_type* rotm(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *y,
    const typename L::tensor_type &p);
template < typename L >
typename L::tensor_type* scal(
    L *l, typename L::tensor_type *x, const typename L::value_type &a);
template < typename L >
typename L::tensor_type* swap(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *y);
template < typename L >
typename L::size_tensor* iamax(
    L *l, const typename L::tensor_type &x, typename L::size_tensor *r);

// BLAS level-2 routines
template < typename L >
const typename L::tensor_type& gbmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::size_type kl,
    typename L::size_type ku);
template < typename L >
const typename L::tensor_type& gemv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta);
template < typename L >
const typename L::tensor_type& ger(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &a, const typename L::value_type &alpha);
template < typename L >
const typename L::tensor_type& gerc(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &a, const typename L::value_type &alpha);
template < typename L >
const typename L::tensor_type& hbmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::size_type k,
    typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& hemv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& her(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &a,
    const typename L::real_type &alpha, typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& her2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &a, const typename L::value_type &alpha,
    typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& hpmv(
    L *l, const typename L::tensor_type &ap, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& hpr(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &ap,
    const typename L::real_type &alpha, typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& hpr2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &ap, const typename L::value_type &alpha,
    typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& sbmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::size_type k,
    typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& spmv(
    L *l, const typename L::tensor_type &ap, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& spr(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &ap,
    const typename L::real_type &alpha, typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& spr2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &ap, const typename L::value_type &alpha,
    typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& symv(
    L *l, const typename L::tensor_type &ap, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& syr(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &a,
    const typename L::real_type &alpha, typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& syr2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &a, const typename L::value_type &alpha,
    typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& tbmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::size_type k, typename L::Uplo uplo,
    typename L::Diag diag);
template < typename L >
const typename L::tensor_type& tbsv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::size_type k, typename L::Uplo uplo,
    typename L::Diag diag);
template < typename L >
const typename L::tensor_type& tpmv(
    L *l, const typename L::tensor_type &ap, const typename L::tensor_type &x,
    typename L::Uplo uplo, typename L::Diag diag);
template < typename L >
const typename L::tensor_type& tpsv(
    L *l, const typename L::tensor_type &ap, const typename L::tensor_type &x,
    typename L::Uplo uplo, typename L::Diag diag);
template < typename L >
const typename L::tensor_type& trmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::Uplo uplo, typename L::Diag diag);
template < typename L >
const typename L::tensor_type& trsv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::Uplo uplo, typename L::Diag diag);
template < typename L >
typename L::tensor_type* gbmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::tensor_type *y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::size_type kl,
    typename L::size_type ku);
template < typename L >
typename L::tensor_type* gemv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::tensor_type *y, const typename L::value_type &alpha,
    const typename L::value_type &beta);
template < typename L >
typename L::tensor_type* ger(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    typename L::tensor_type *a, const typename L::value_type &alpha);
template < typename L >
typename L::tensor_type* gerc(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    typename L::tensor_type *a, const typename L::value_type &alpha);
template < typename L >
typename L::tensor_type* hbmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::tensor_type *y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::size_type k,
    typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* hemv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::tensor_type *y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* her(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *a,
    const typename L::real_type &alpha, typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* her2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    typename L::tensor_type *a, const typename L::value_type &alpha,
    typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* hpmv(
    L *l, const typename L::tensor_type &ap, const typename L::tensor_type &x,
    typename L::tensor_type *y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* hpr(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *ap,
    const typename L::real_type &alpha, typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* hpr2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    typename L::tensor_type *ap, const typename L::value_type &alpha,
    typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* sbmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::tensor_type *y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::size_type k,
    typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* spmv(
    L *l, const typename L::tensor_type &ap, const typename L::tensor_type &x,
    typename L::tensor_type *y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* spr(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *ap,
    const typename L::real_type &alpha, typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* spr2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    typename L::tensor_type *ap, const typename L::value_type &alpha,
    typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* symv(
    L *l, const typename L::tensor_type &p, const typename L::tensor_type &x,
    typename L::tensor_type *y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* syr(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *a,
    const typename L::real_type &alpha, typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* syr2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    typename L::tensor_type *a, const typename L::value_type &alpha,
    typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* tbmv(
    L *l, const typename L::tensor_type &a, typename L::tensor_type *x,
    typename L::size_type k, typename L::Uplo uplo,
    typename L::Diag diag);
template < typename L >
typename L::tensor_type* tbsv(
    L *l, const typename L::tensor_type &a, typename L::tensor_type *x,
    typename L::size_type k, typename L::Uplo uplo,
    typename L::Diag diag);
template < typename L >
typename L::tensor_type* tpmv(
    L *l, const typename L::tensor_type &ap, typename L::tensor_type *x,
    typename L::Uplo uplo, typename L::Diag diag);
template < typename L >
typename L::tensor_type* tpsv(
    L *l, const typename L::tensor_type &ap, typename L::tensor_type *x,
    typename L::Uplo uplo, typename L::Diag diag);
template < typename L >
typename L::tensor_type* trmv(
    L *l, const typename L::tensor_type &a, typename L::tensor_type *x,
    typename L::Uplo uplo, typename L::Diag diag);
template < typename L >
typename L::tensor_type* trsv(
    L *l, const typename L::tensor_type &a, typename L::tensor_type *x,
    typename L::Uplo uplo, typename L::Diag diag);

// BLAS level-3 routines
template < typename L >
const typename L::tensor_type& gemm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &b,
    const typename L::tensor_type &c, const typename L::value_type &alpha,
    const typename L::value_type &beta);
template < typename L >
const typename L::tensor_type& hemm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &b,
    const typename L::tensor_type &c, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Side side,
    typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& herk(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &c,
    const typename L::real_type &alpha, const typename L::real_type &beta,
    typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& her2k(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &b,
    const typename L::tensor_type &c, const typename L::value_type &alpha,
    const typename L::real_type &beta, typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& symm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &b,
    const typename L::tensor_type &c, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Side side,
    typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& syrk(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &c,
    const typename L::value_type &alpha, const typename L::value_type &beta,
    typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& syr2k(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &b,
    const typename L::tensor_type &c, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo);
template < typename L >
const typename L::tensor_type& trmm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &b,
    const typename L::value_type &alpha, typename L::Side side,
    typename L::Uplo uplo, typename L::Diag diag);
template < typename L >
const typename L::tensor_type& trsm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &b,
    const typename L::value_type &alpha, typename L::Side side,
    typename L::Uplo uplo, typename L::Diag diag);
template < typename L >
typename L::tensor_type* gemm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &b,
    typename L::tensor_type *c, const typename L::value_type &alpha,
    const typename L::value_type &beta);
template < typename L >
typename L::tensor_type* hemm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &b,
    typename L::tensor_type *c, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Side side,
    typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* herk(
    L *l, const typename L::tensor_type &a, typename L::tensor_type *c,
    const typename L::real_type &alpha, const typename L::real_type &beta,
    typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* her2k(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &b,
    typename L::tensor_type *c, const typename L::value_type &alpha,
    const typename L::real_type &beta, typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* symm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &b,
    typename L::tensor_type *c, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Side side,
    typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* syrk(
    L *l, const typename L::tensor_type &a, typename L::tensor_type *c,
    const typename L::value_type &alpha, const typename L::value_type &beta,
    typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* syr2k(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &b,
    typename L::tensor_type *c, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo);
template < typename L >
typename L::tensor_type* trmm(
    L *l, const typename L::tensor_type &a, typename L::tensor_type *b,
    const typename L::value_type &alpha, typename L::Side side,
    typename L::Uplo uplo, typename L::Diag diag);
template < typename L >
typename L::tensor_type* trsm(
    L *l, const typename L::tensor_type &a, typename L::tensor_type *b,
    const typename L::value_type &alpha, typename L::Side side,
    typename L::Uplo uplo, typename L::Diag diag);

}  // namespace math
}  // namespace linalg
}  // namespace thunder

#endif  // THUNDER_LINALG_MATH_HPP_
