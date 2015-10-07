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
    const T &a, const T &x, const T &y, size_type kl, size_type ku,
    const value_type &alpha, const value_type &beta) {
  return math::gbmv(this, a, x, y, kl, ku, alpha, beta);
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
    const T &a, const T &x, const T &y, const value_type &alpha, size_type k,
    const value_type &beta, Uplo uplo) {
  return math::hbmv(this, a, x, y, alpha, k, beta, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::hemv(
    const T &a, const T &x, const T &y, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return math::hemv(this, a, x, y, alpha, beta, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::her(
    const T &x, const T &a, const value_type &alpha, Uplo uplo) {
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
    const T &x, const T &ap, const value_type &alpha, Uplo uplo) {
  return math::hpr(this, x, ap, alpha, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::hpr2(
    const T &x, const T &y, const T &ap, const value_type &alpha, Uplo uplo) {
  return math::hpr2(this, x, y, ap, alpha, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::sbmv(
    const T &a, const T &x, const T &y, size_type k, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return math::sbmv(this, a, x, y, k, alpha, beta, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::spmv(
    const T &ap, const T &x, const T &y, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return math::spmv(this, ap, x, y, alpha, beta, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::spr(
    const T &x, const T &ap, const value_type &alpha, Uplo uplo) {
  return math::spr(this, x, ap, alpha, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::spr2(
    const T &x, const T &y, const T &ap, const value_type &alpha, Uplo uplo) {
  return math::spr2(this, x, y, ap, alpha, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::symv(
    const T &ap, const T &x, const T &y, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return math::symv(this, ap, x, y, alpha, beta, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::syr(
    const T &x, const T &a, const value_type &alpha, Uplo uplo) {
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


}  //  namespace linalg
}  //  namespace thunder

#endif  // THUNDER_LINALG_LINALG_INL_BLASLEVEL2_HPP_
