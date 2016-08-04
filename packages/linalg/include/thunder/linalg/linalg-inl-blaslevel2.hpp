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

#ifndef THUNDER_LINALG_LINALG_INL_BLASLEVEL2_HPP_
#define THUNDER_LINALG_LINALG_INL_BLASLEVEL2_HPP_

#include "thunder/linalg/linalg.hpp"
#include "thunder/linalg/linalg-inl.hpp"

#include "thunder/linalg/math.hpp"

namespace thunder {
namespace linalg {

template < typename T, typename H >
const T& Linalg< T, H >::gbmv(
    const T &a, const T &x, const T &y, const value_type &alpha,
    const value_type &beta, size_type kl, size_type ku) {
  return math::gbmv(this, a, x, y, alpha, beta, kl, ku);
}

template < typename T, typename H >
const T& Linalg< T, H >::gemv(
    const T &a, const T &x, const T &y, const value_type &alpha,
    const value_type &beta) {
  return math::gemv(this, a, x, y, alpha, beta);
}

template < typename T, typename H >
const T& Linalg< T, H >::ger(
    const T &x, const T &y, const T &a, const value_type &alpha) {
  return math::ger(this, x, y, a, alpha);
}

template < typename T, typename H >
const T& Linalg< T, H >::gerc(
    const T &x, const T &y, const T &a, const value_type &alpha) {
  return math::gerc(this, x, y, a, alpha);
}

template < typename T, typename H >
const T& Linalg< T, H >::hbmv(
    const T &a, const T &x, const T &y, const value_type &alpha,
    const value_type &beta, size_type k, Uplo uplo) {
  return math::hbmv(this, a, x, y, alpha, beta, k, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::hemv(
    const T &a, const T &x, const T &y, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return math::hemv(this, a, x, y, alpha, beta, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::her(
    const T &x, const T &a, const real_type &alpha, Uplo uplo) {
  return math::her(this, x, a, alpha, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::her2(
    const T &x, const T &y, const T &a, const value_type &alpha, Uplo uplo) {
  return math::her2(this, x, y, a, alpha, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::hpmv(
    const T &ap, const T &x, const T &y, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return math::hpmv(this, ap, x, y, alpha, beta, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::hpr(
    const T &x, const T &ap, const real_type &alpha, Uplo uplo) {
  return math::hpr(this, x, ap, alpha, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::hpr2(
    const T &x, const T &y, const T &ap, const value_type &alpha, Uplo uplo) {
  return math::hpr2(this, x, y, ap, alpha, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::sbmv(
    const T &a, const T &x, const T &y, const value_type &alpha,
    const value_type &beta, size_type k, Uplo uplo) {
  return math::sbmv(this, a, x, y, alpha, beta, k, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::spmv(
    const T &ap, const T &x, const T &y, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return math::spmv(this, ap, x, y, alpha, beta, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::spr(
    const T &x, const T &ap, const real_type &alpha, Uplo uplo) {
  return math::spr(this, x, ap, alpha, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::spr2(
    const T &x, const T &y, const T &ap, const value_type &alpha, Uplo uplo) {
  return math::spr2(this, x, y, ap, alpha, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::symv(
    const T &a, const T &x, const T &y, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return math::symv(this, a, x, y, alpha, beta, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::syr(
    const T &x, const T &a, const real_type &alpha, Uplo uplo) {
  return math::syr(this, x, a, alpha, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::syr2(
    const T &x, const T &y, const T &a, const value_type &alpha, Uplo uplo) {
  return math::syr2(this, x, y, a, alpha, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::tbmv(
    const T &a, const T &x, size_type k, Uplo uplo, Diag diag) {
  return math::tbmv(this, a, x, k, uplo, diag);
}

template < typename T, typename H >
const T& Linalg< T, H >::tbsv(
    const T &a, const T &x, size_type k, Uplo uplo, Diag diag) {
  return math::tbsv(this, a, x, k, uplo, diag);
}

template < typename T, typename H >
const T& Linalg< T, H >::tpmv(const T &ap, const T &x, Uplo uplo, Diag diag) {
  return math::tpmv(this, ap, x, uplo, diag);
}

template < typename T, typename H >
const T& Linalg< T, H >::tpsv(const T &ap, const T &x, Uplo uplo, Diag diag) {
  return math::tpsv(this, ap, x, uplo, diag);
}

template < typename T, typename H >
const T& Linalg< T, H >::trmv(const T &a, const T &x, Uplo uplo, Diag diag) {
  return math::trmv(this, a, x, uplo, diag);
}

template < typename T, typename H >
const T& Linalg< T, H >::trsv(const T &a, const T &x, Uplo uplo, Diag diag) {
  return math::trsv(this, a, x, uplo, diag);
}

template < typename T, typename H >
T& Linalg<T, H>::gbmv(
    const T &a, const T &x, T &y, const value_type &alpha,
    const value_type &beta, size_type kl, size_type ku) {
  return const_cast< T& >(gbmv(
      a, x, const_cast< const T& >(y), alpha, beta, kl, ku));
}

template < typename T, typename H >
T& Linalg<T, H>::gemv(
    const T &a, const T &x, T &y, const value_type &alpha,
    const value_type &beta) {
  return const_cast< T& >(gemv(
      a, x, const_cast< const T& >(y), alpha, beta));
}

template < typename T, typename H >
T& Linalg<T, H>::ger(const T &x, const T &y, T &a, const value_type &alpha) {
  return const_cast< T& >(ger(x, y, const_cast< const T& >(a), alpha));
}

template < typename T, typename H >
T& Linalg<T, H>::gerc(const T &x, const T &y, T &a, const value_type &alpha) {
  return const_cast< T& >(gerc(x, y, const_cast< const T& >(a), alpha));
}

template < typename T, typename H >
T& Linalg<T, H>::hbmv(
    const T &a, const T &x, T &y, const value_type &alpha,
    const value_type &beta, size_type k, Uplo uplo) {
  return const_cast< T& >(hbmv(
      a, x, const_cast< const T& >(y), alpha, beta, k, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::hemv(
    const T &a, const T &x, T &y, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return const_cast< T& >(hemv(
      a, x, const_cast< const T& >(y), alpha, beta, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::her(const T &x, T &a, const real_type &alpha, Uplo uplo) {
  return const_cast< T& >(her(
      x, const_cast< const T& >(a), alpha, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::her2(
    const T &x, const T &y, T &a, const value_type &alpha, Uplo uplo) {
  return const_cast< T& >(her2(
      x, y, const_cast< const T& >(a), alpha, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::hpmv(
    const T &ap, const T &x, T &y, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return const_cast< T& >(hpmv(
      ap, x, const_cast< const T& >(y), alpha, beta, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::hpr(const T &x, T &ap, const real_type &alpha, Uplo uplo) {
  return const_cast< T& >(hpr(
      x, const_cast< const T& >(ap), alpha, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::hpr2(
    const T &x, const T &y, T &ap, const value_type &alpha, Uplo uplo) {
  return const_cast< T& >(hpr2(
      x, y, const_cast< const T& >(ap), alpha, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::sbmv(
    const T &a, const T &x, T &y, const value_type &alpha,
    const value_type &beta, size_type k, Uplo uplo) {
  return const_cast< T& >(sbmv(
      a, x, const_cast< const T& >(y), alpha, beta, k, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::spmv(
    const T &ap, const T &x, T &y, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return const_cast< T& >(spmv(
      ap, x, const_cast< const T& >(y), alpha, beta, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::spr(
    const T &x, T &ap, const real_type &alpha, Uplo uplo) {
  return const_cast< T& >(spr(x, const_cast< const T& >(ap), alpha, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::spr2(const T &x, const T &y, T &ap,
                      const value_type &alpha, Uplo uplo) {
  return const_cast< T& >(spr2(x, y, const_cast< const T& >(ap), alpha, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::symv(
    const T &a, const T &x, T &y, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return const_cast< T& >(symv(
      a, x, const_cast< const T& >(y), alpha, beta, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::syr(const T &x, T &a, const real_type &alpha, Uplo uplo) {
  return const_cast< T& >(syr(
      x, const_cast< const T& >(a), alpha, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::syr2(
    const T &x, const T &y, T &a, const value_type &alpha, Uplo uplo) {
  return const_cast< T& >(syr2(x, y, const_cast< const T& >(a), alpha, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::tbmv(const T &a, T &x, size_type k, Uplo uplo, Diag diag) {
  return const_cast< T& >(tbmv(a, const_cast< const T& >(x), k, uplo, diag));
}

template < typename T, typename H >
T& Linalg<T, H>::tbsv(const T &a, T &x, size_type k, Uplo uplo, Diag diag) {
  return const_cast< T& >(tbsv(a, const_cast< const T& >(x), k, uplo, diag));
}

template < typename T, typename H >
T& Linalg<T, H>::tpmv(const T &ap, T &x, Uplo uplo, Diag diag) {
  return const_cast< T& >(tpmv(ap, const_cast< const T& >(x), uplo, diag));
}

template < typename T, typename H >
T& Linalg<T, H>::tpsv(const T &ap, T &x, Uplo uplo, Diag diag) {
  return const_cast< T& >(tpsv(ap, const_cast< const T& >(x), uplo, diag));
}

template < typename T, typename H >
T& Linalg<T, H>::trmv(const T &a, T &x, Uplo uplo, Diag diag) {
  return const_cast< T& >(trmv(a, const_cast< const T& >(x), uplo, diag));
}

template < typename T, typename H >
T& Linalg<T, H>::trsv(const T &a, T &x, Uplo uplo, Diag diag) {
  return const_cast< T& >(trsv(a, const_cast< const T& >(x), uplo, diag));
}

template < typename T, typename H >
T* Linalg< T, H >::gbmv(
    const T &a, const T &x, T *y, const value_type &alpha,
    const value_type &beta, size_type kl, size_type ku) {
  return math::gbmv(this, a, x, y, alpha, beta, kl, ku);
}

template < typename T, typename H >
T* Linalg< T, H >::gemv(
    const T &a, const T &x, T *y, const value_type &alpha,
    const value_type &beta) {
  return math::gemv(this, a, x, y, alpha, beta);
}

template < typename T, typename H >
T* Linalg< T, H >::ger(
    const T &x, const T &y, T *a, const value_type &alpha) {
  return math::ger(this, x, y, a, alpha);
}

template < typename T, typename H >
T* Linalg< T, H >::gerc(
    const T &x, const T &y, T *a, const value_type &alpha) {
  return math::gerc(this, x, y, a, alpha);
}

template < typename T, typename H >
T* Linalg< T, H >::hbmv(
    const T &a, const T &x, T *y, const value_type &alpha,
    const value_type &beta, size_type k, Uplo uplo) {
  return math::hbmv(this, a, x, y, alpha, beta, k, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::hemv(
    const T &a, const T &x, T *y, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return math::hemv(this, a, x, y, alpha, beta, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::her(
    const T &x, T *a, const real_type &alpha, Uplo uplo) {
  return math::her(this, x, a, alpha, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::her2(
    const T &x, const T &y, T *a, const value_type &alpha, Uplo uplo) {
  return math::her2(this, x, y, a, alpha, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::hpmv(
    const T &ap, const T &x, T *y, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return math::hpmv(this, ap, x, y, alpha, beta, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::hpr(
    const T &x, T *ap, const real_type &alpha, Uplo uplo) {
  return math::hpr(this, x, ap, alpha, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::hpr2(
    const T &x, const T &y, T *ap, const value_type &alpha, Uplo uplo) {
  return math::hpr2(this, x, y, ap, alpha, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::sbmv(
    const T &a, const T &x, T *y, const value_type &alpha,
    const value_type &beta, size_type k, Uplo uplo) {
  return math::sbmv(this, a, x, y, alpha, beta, k, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::spmv(
    const T &ap, const T &x, T *y, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return math::spmv(this, ap, x, y, alpha, beta, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::spr(
    const T &x, T *ap, const real_type &alpha, Uplo uplo) {
  return math::spr(this, x, ap, alpha, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::spr2(
    const T &x, const T &y, T *ap, const value_type &alpha, Uplo uplo) {
  return math::spr2(this, x, y, ap, alpha, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::symv(
    const T &a, const T &x, T *y, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return math::symv(this, a, x, y, alpha, beta, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::syr(
    const T &x, T *a, const real_type &alpha, Uplo uplo) {
  return math::syr(this, x, a, alpha, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::syr2(
    const T &x, const T &y, T *a, const value_type &alpha, Uplo uplo) {
  return math::syr2(this, x, y, a, alpha, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::tbmv(
    const T &a, T *x, size_type k, Uplo uplo, Diag diag) {
  return math::tbmv(this, a, x, k, uplo, diag);
}

template < typename T, typename H >
T* Linalg< T, H >::tbsv(
    const T &a, T *x, size_type k, Uplo uplo, Diag diag) {
  return math::tbsv(this, a, x, k, uplo, diag);
}

template < typename T, typename H >
T* Linalg< T, H >::tpmv(const T &ap, T *x, Uplo uplo, Diag diag) {
  return math::tpmv(this, ap, x, uplo, diag);
}

template < typename T, typename H >
T* Linalg< T, H >::tpsv(const T &ap, T *x, Uplo uplo, Diag diag) {
  return math::tpsv(this, ap, x, uplo, diag);
}

template < typename T, typename H >
T* Linalg< T, H >::trmv(const T &a, T *x, Uplo uplo, Diag diag) {
  return math::trmv(this, a, x, uplo, diag);
}

template < typename T, typename H >
T* Linalg< T, H >::trsv(const T &a, T *x, Uplo uplo, Diag diag) {
  return math::trsv(this, a, x, uplo, diag);
}

template < typename T, typename H >
T Linalg<T, H>::gbmv(
    const T &a, const T &x, const value_type &alpha, const value_type &beta,
    size_type kl, size_type ku) {
  T y(a.allocator());
  gbmv(a, x, &y, alpha, beta, kl, ku);
  return y;
}

template < typename T, typename H >
T Linalg<T, H>::gemv(
    const T &a, const T &x, const value_type &alpha, const value_type &beta) {
  T y(a.allocator());
  gemv(a, x, &y, alpha, beta);
  return y;
}

template < typename T, typename H >
T Linalg<T, H>::ger(const T &x, const T &y, const value_type &alpha) {
  T a(x.allocator());
  ger(x, y, &a, alpha);
  return a;
}

template < typename T, typename H >
T Linalg<T, H>::gerc(const T &x, const T &y, const value_type &alpha) {
  T a(x.allocator());
  gerc(x, y, &a, alpha);
  return a;
}

template < typename T, typename H >
T Linalg<T, H>::hbmv(
    const T &a, const T &x, const value_type &alpha, const value_type &beta,
    size_type k, Uplo uplo) {
  T y(a.allocator());
  hbmv(a, x, &y, alpha, beta, k, uplo);
  return y;
}

template < typename T, typename H >
T Linalg<T, H>::hemv(
    const T &a, const T &x, const value_type &alpha, const value_type &beta,
    Uplo uplo) {
  T y(a.allocator());
  hemv(a, x, &y, alpha, beta, uplo);
  return y;
}

template < typename T, typename H >
T Linalg<T, H>::her(const T &x, const real_type &alpha, Uplo uplo) {
  T a(x.allocator());
  her(x, &a, alpha, uplo);
  return a;
}

template < typename T, typename H >
T Linalg<T, H>::her2(
    const T &x, const T &y, const value_type &alpha, Uplo uplo) {
  T a(x.allocator());
  her2(x, y, &a, alpha, uplo);
  return a;
}

template < typename T, typename H >
T Linalg<T, H>::hpmv(
    const T &ap, const T &x, const value_type &alpha, const value_type &beta,
    Uplo uplo) {
  T y(ap.allocator());
  hpmv(ap, x, &y, alpha, beta, uplo);
  return y;
}

template < typename T, typename H >
T Linalg<T, H>::hpr(const T &x, const real_type &alpha, Uplo uplo) {
  T ap(x.allocator());
  hpr(x, &ap, alpha, uplo);
  return ap;
}

template < typename T, typename H >
T Linalg<T, H>::hpr2(
    const T &x, const T &y, const value_type &alpha, Uplo uplo) {
  T ap(x.allocator());
  hpr2(x, y, &ap, alpha, uplo);
  return ap;
}

template < typename T, typename H >
T Linalg<T, H>::sbmv(
    const T &a, const T &x, const value_type &alpha, const value_type &beta,
    size_type k, Uplo uplo) {
  T y(a.allocator());
  sbmv(a, x, &y, alpha, beta, k, uplo);
  return y;
}

template < typename T, typename H >
T Linalg<T, H>::spmv(
    const T &ap, const T &x, const value_type &alpha, const value_type &beta,
    Uplo uplo) {
  T y(ap.allocator());
  spmv(ap, x, &y, alpha, beta, uplo);
  return y;
}

template < typename T, typename H >
T Linalg<T, H>::spr(const T &x, const real_type &alpha, Uplo uplo) {
  T ap(x.allocator());
  spr(x, &ap, alpha, uplo);
  return ap;
}

template < typename T, typename H >
T Linalg<T, H>::spr2(
    const T &x, const T &y, const value_type &alpha, Uplo uplo) {
  T ap(x.allocator());
  spr2(x, y, &ap, alpha, uplo);
  return ap;
}

template < typename T, typename H >
T Linalg<T, H>::symv(
    const T &a, const T &x, const value_type &alpha, const value_type &beta,
    Uplo uplo) {
  T y(a.allocator());
  symv(a, x, &y, alpha, beta, uplo);
  return y;
}

template < typename T, typename H >
T Linalg<T, H>::syr(const T &x, const real_type &alpha, Uplo uplo) {
  T a(x.allocator());
  syr(x, &a, alpha, uplo);
  return a;
}

template < typename T, typename H >
T Linalg<T, H>::syr2(
    const T &x, const T &y, const value_type &alpha, Uplo uplo) {
  T a(x.allocator());
  syr2(x, y, &a, alpha, uplo);
  return a;
}

template < typename T, typename H >
T Linalg<T, H>::tbmv(const T &a, size_type k, Uplo uplo, Diag diag) {
  T x(a.allocator());
  tbmv(a, &x, k, uplo, diag);
  return x;
}

template < typename T, typename H >
T Linalg<T, H>::tbsv(const T &a, size_type k, Uplo uplo, Diag diag) {
  T x(a.allocator());
  tbsv(a, &x, k, uplo, diag);
  return x;
}

template < typename T, typename H >
T Linalg<T, H>::tpmv(const T &ap, Uplo uplo, Diag diag) {
  T x(ap.allocator());
  tpmv(ap, &x, uplo, diag);
  return x;
}

template < typename T, typename H >
T Linalg<T, H>::tpsv(const T &ap, Uplo uplo, Diag diag) {
  T x(ap.allocator());
  tpsv(ap, &x, uplo, diag);
  return x;
}

template < typename T, typename H >
T Linalg<T, H>::trmv(const T &a, Uplo uplo, Diag diag) {
  T x(a.allocator());
  trmv(a, &x, uplo, diag);
  return x;
}

template < typename T, typename H >
T Linalg<T, H>::trsv(const T &a, Uplo uplo, Diag diag) {
  T x(a.allocator());
  trsv(a, &x, uplo, diag);
  return x;
}

}  //  namespace linalg
}  //  namespace thunder

#endif  // THUNDER_LINALG_LINALG_INL_BLASLEVEL2_HPP_
