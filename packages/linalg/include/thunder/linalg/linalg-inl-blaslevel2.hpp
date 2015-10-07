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

}  //  namespace linalg
}  //  namespace thunder

#endif  // THUNDER_LINALG_LINALG_INL_BLASLEVEL2_HPP_
