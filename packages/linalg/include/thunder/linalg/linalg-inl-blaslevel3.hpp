/*
 * \copyright Copyright 2014-2016 Xiang Zhang All Rights Reserved.
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

#ifndef THUNDER_LINALG_LINALG_INL_BLASLEVEL3_HPP_
#define THUNDER_LINALG_LINALG_INL_BLASLEVEL3_HPP_

#include "thunder/linalg/linalg.hpp"
#include "thunder/linalg/linalg-inl.hpp"

#include "thunder/linalg/math.hpp"

namespace thunder {
namespace linalg {

template < typename T, typename H >
const T& Linalg< T, H >::gemm(
    const T &a, const T &b, const T &c, const value_type &alpha,
    const value_type &beta) {
  return math::gemm(this, a, b, c, alpha, beta);
}

template < typename T, typename H >
const T& Linalg< T, H >::hemm(
    const T &a, const T &b, const T &c, const value_type &alpha,
    const value_type &beta, Side side, Uplo uplo) {
  return math::hemm(this, a, b, c, alpha, beta, side, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::herk(
    const T &a, const T &c, const real_type &alpha, const real_type &beta,
    Uplo uplo) {
  return math::herk(this, a, c, alpha, beta, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::her2k(
    const T &a, const T &b, const T &c, const value_type &alpha,
    const real_type &beta, Uplo uplo) {
  return math::her2k(this, a, b, c, alpha, beta, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::symm(
    const T &a, const T &b, const T &c, const value_type &alpha,
    const value_type &beta, Side side, Uplo uplo) {
  return math::symm(this, a, b, c, alpha, beta, side, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::syrk(
    const T &a, const T &c, const value_type &alpha, const value_type &beta,
    Uplo uplo) {
  return math::syrk(this, a, c, alpha, beta, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::syr2k(
    const T &a, const T &b, const T &c, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return math::syr2k(this, a, b, c, alpha, beta, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::trmm(
    const T &a, const T &b, const value_type &alpha, Side side, Uplo uplo,
    Diag diag) {
  return math::trmm(this, a, b, alpha, side, uplo, diag);
}

template < typename T, typename H >
const T& Linalg< T, H >::trsm(
    const T &a, const T &b, const value_type &alpha, Side side, Uplo uplo,
    Diag diag) {
  return math::trsm(this, a, b, alpha, side, uplo, diag);
}

template < typename T, typename H >
T& Linalg<T, H>::gemm(
    const T &a, const T &b, T &c, const value_type &alpha,
    const value_type &beta) {
  return const_cast< T& >(gemm(a, b, const_cast< const T& >(c), alpha, beta));
}

template < typename T, typename H >
T& Linalg<T, H>::hemm(
    const T &a, const T &b, T &c, const value_type &alpha,
    const value_type &beta, Side side, Uplo uplo) {
  return const_cast< T& >(hemm(
      a, b, const_cast< const T& >(c), alpha, beta, side, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::herk(
    const T &a, T &c, const real_type &alpha, const real_type &beta,
    Uplo uplo) {
  return const_cast< T& >(herk(
      a, const_cast< const T& >(c), alpha, beta, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::her2k(
    const T &a, const T &b, T &c, const value_type &alpha,
    const real_type &beta, Uplo uplo) {
  return const_cast< T& >(her2k(
      a, b, const_cast< const T& >(c), alpha, beta, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::symm(
    const T &a, const T &b, T &c, const value_type &alpha,
    const value_type &beta, Side side, Uplo uplo) {
  return const_cast< T& >(symm(
      a, b, const_cast< const T& >(c), alpha, beta, side, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::syrk(
    const T &a, T &c, const value_type &alpha, const value_type &beta,
    Uplo uplo) {
  return const_cast< T& >(syrk(
      a, const_cast< const T& >(c), alpha, beta, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::syr2k(
    const T &a, const T &b, T &c, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return const_cast< T& >(syr2k(
      a, b, const_cast< const T& >(c), alpha, beta, uplo));
}

template < typename T, typename H >
T& Linalg<T, H>::trmm(
    const T &a, T &b, const value_type &alpha, Side side, Uplo uplo,
    Diag diag) {
  return const_cast< T& >(trmm(
      a, const_cast< const T& >(b), alpha, side, uplo, diag));
}

template < typename T, typename H >
T& Linalg<T, H>::trsm(
    const T &a, T &b, const value_type &alpha, Side side, Uplo uplo,
    Diag diag) {
  return const_cast< T& >(trsm(
      a, const_cast< const T& >(b), alpha, side, uplo, diag));
}

template < typename T, typename H >
T* Linalg< T, H >::gemm(
    const T &a, const T &b, T *c, const value_type &alpha,
    const value_type &beta) {
  return math::gemm(this, a, b, c, alpha, beta);
}

template < typename T, typename H >
T* Linalg< T, H >::hemm(
    const T &a, const T &b, T *c, const value_type &alpha,
    const value_type &beta, Side side, Uplo uplo) {
  return math::hemm(this, a, b, c, alpha, beta, side, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::herk(
    const T &a, T *c, const real_type &alpha, const real_type &beta,
    Uplo uplo) {
  return math::herk(this, a, c, alpha, beta, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::her2k(
    const T &a, const T &b, T *c, const value_type &alpha,
    const real_type &beta, Uplo uplo) {
  return math::her2k(this, a, b, c, alpha, beta, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::symm(
    const T &a, const T &b, T *c, const value_type &alpha,
    const value_type &beta, Side side, Uplo uplo) {
  return math::symm(this, a, b, c, alpha, beta, side, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::syrk(
    const T &a, T *c, const value_type &alpha, const value_type &beta,
    Uplo uplo) {
  return math::syrk(this, a, c, alpha, beta, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::syr2k(
    const T &a, const T&b, T *c, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return math::syr2k(this, a, b, c, alpha, beta, uplo);
}

template < typename T, typename H >
T* Linalg< T, H >::trmm(
    const T &a, T *b, const value_type &alpha, Side side, Uplo uplo,
    Diag diag) {
  return math::trmm(this, a, b, alpha, side, uplo, diag);
}

template < typename T, typename H >
T* Linalg< T, H >::trsm(
    const T &a, T *b, const value_type &alpha, Side side, Uplo uplo,
    Diag diag) {
  return math::trsm(this, a, b, alpha, side, uplo, diag);
}

template < typename T, typename H >
T Linalg<T, H>::gemm(
    const T &a, const T &b, const value_type &alpha, const value_type &beta) {
  T c(a.allocator());
  gemm(a, b, &c, alpha, beta);
  return c;
}

template < typename T, typename H >
T Linalg<T, H>::hemm(
    const T &a, const T &b, const value_type &alpha, const value_type &beta,
    Side side, Uplo uplo) {
  T c(a.allocator());
  hemm(a, b, &c, alpha, beta, side, uplo);
  return c;
}

template < typename T, typename H >
T Linalg<T, H>::herk(
    const T &a, const real_type &alpha, const real_type &beta, Uplo uplo) {
  T c(a.allocator());
  herk(a, &c, alpha, beta, uplo);
  return c;
}

template < typename T, typename H >
T Linalg<T, H>::her2k(
    const T &a, const T &b, const value_type &alpha, const real_type &beta,
    Uplo uplo) {
  T c(a.allocator());
  her2k(a, b, &c, alpha, beta, uplo);
  return c;
}

template < typename T, typename H >
T Linalg<T, H>::symm(
    const T &a, const T &b, const value_type &alpha, const value_type &beta,
    Side side, Uplo uplo) {
  T c(a.allocator());
  symm(a, b, &c, alpha, beta, side, uplo);
  return c;
}

template < typename T, typename H >
T Linalg<T, H>::syrk(
    const T &a, const value_type &alpha, const value_type &beta, Uplo uplo) {
  T c(a.allocator());
  syrk(a, &c, alpha, beta, uplo);
  return c;
}

template < typename T, typename H >
T Linalg<T, H>::syr2k(
    const T &a, const T &b, const value_type &alpha, const value_type &beta,
    Uplo uplo) {
  T c(a.allocator());
  syr2k(a, b, &c, alpha, beta, uplo);
  return c;
}

template < typename T, typename H >
T Linalg<T, H>::trmm(
    const T &a, const value_type &alpha, Side side, Uplo uplo, Diag diag) {
  T b(a.allocator());
  trmm(a, &b, alpha, side, uplo, diag);
  return b;
}

template < typename T, typename H >
T Linalg<T, H>::trsm(
    const T &a, const value_type &alpha, Side side, Uplo uplo, Diag diag) {
  T b(a.allocator());
  trsm(a, &b, alpha, side, uplo, diag);
  return b;
}

}  //  namespace linalg
}  //  namespace thunder

#endif  // THUNDER_LINALG_LINALG_INL_BLASLEVEL3_HPP_
