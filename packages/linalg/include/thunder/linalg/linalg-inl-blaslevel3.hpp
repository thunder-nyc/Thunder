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
    const value_type &beta, Uplo uplo) {
  return math::hemm(this, a, b, c, alpha, beta, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::herk(
    const T &a, const T &c, const value_type &alpha, const value_type &beta,
    Uplo uplo) {
  return math::herk(this, a, c, alpha, beta, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::herk2(
    const T &a, const T &b, const T &c, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return math::herk2(this, a, b, c, alpha, beta, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::symm(
    const T &a, const T &b, const T &c, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return math::symm(this, a, b, c, alpha, beta, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::syrk(
    const T &a, const T &c, const value_type &alpha, const value_type &beta,
    Uplo uplo) {
  return math::syrk(this, a, c, alpha, beta, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::syrk2(
    const T &a, const T&b, const T &c, const value_type &alpha,
    const value_type &beta, Uplo uplo) {
  return math::syrk2(this, a, b, c, alpha, beta, uplo);
}

template < typename T, typename H >
const T& Linalg< T, H >::trmm(
    const T &a, const T&b, const T &c, const value_type &alpha,
    const value_type &beta, Uplo uplo, Diag diag) {
  return math::trmm(this, a, b, c, alpha, beta, uplo, diag);
}

template < typename T, typename H >
const T& Linalg< T, H >::trsm(
    const T &a, const T &b, const value_type &alpha, Side side, Uplo uplo,
    Diag diag) {
  return math::trsm(this, a, b, alpha, side, uplo, diag);
}

}  //  namespace linalg
}  //  namespace thunder

#endif  // THUNDER_LINALG_LINALG_INL_BLASLEVEL3_HPP_
