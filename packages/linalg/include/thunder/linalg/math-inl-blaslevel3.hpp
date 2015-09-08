/*
 * \copyright Copyright 2015 Xiang Zhang All Rights Reserved.
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

#ifndef THUNDER_LINALG_MATH_INL_BLASLEVEL1_HPP_
#define THUNDER_LINALG_MATH_INL_BLASLEVEL1_HPP_

#include "thunder/linalg/math.hpp"
#include "thunder/linalg/math-inl.hpp"

#include "thunder/tensor.hpp"

namespace thunder {
namespace linalg {
namespace math {

template < typename L >
const typename L::tensor_type& gemm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type&b,
    const typename L::tensor_type &c, const typename L::value_type &alpha,
    const typename L::value_type &beta) {
}

template < typename L >
const typename L::tensor_type& hemm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type&b,
    const typename L::tensor_type &c, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::BlasUplo uplo) {
}

template < typename L >
const typename L::tensor_type& herk(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &c,
    const typename L::value_type &alpha, const typename L::value_type &beta,
    typename L::BlasUplo uplo) {
}

template < typename L >
const typename L::tensor_type& herk2(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type&b,
    const typename L::tensor_type &c, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::BlasUplo uplo) {
}

template < typename L >
const typename L::tensor_type& symm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type&b,
    const typename L::tensor_type &c, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::BlasUplo uplo) {
}

template < typename L >
const typename L::tensor_type& syrk(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &c,
    const typename L::value_type &alpha, const typename L::value_type &beta,
    typename L::BlasUplo uplo) {
}

template < typename L >
const typename L::tensor_type& syrk2(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type&b,
    const typename L::tensor_type &c, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::BlasUplo uplo) {
}

template < typename L >
const typename L::tensor_type& trmm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type&b,
    const typename L::tensor_type &c, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::BlasUplo uplo,
    typename L::BlasDiag diag) {
}

template < typename L >
const typename L::tensor_type& trsm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &b,
    const typename L::value_type &alpha, typename L::BlasSide side,
    typename L::BlasUplo uplo, typename L::BlasDiag diag) {
}

}  // namespace math
}  // namespace linalg
}  // namespace thunder

#endif  // THUNDER_LINALG_MATH_INL_BLASLEVEL3_HPP_
