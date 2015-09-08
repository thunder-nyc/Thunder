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

#ifndef THUNDER_LINALG_MATH_INL_BLASLEVEL2_HPP_
#define THUNDER_LINALG_MATH_INL_BLASLEVEL2_HPP_

#include "thunder/linalg/math.hpp"
#include "thunder/linalg/math-inl.hpp"

#include "thunder/tensor.hpp"

namespace thunder {
namespace linalg {
namespace math {

template < typename L >
const typename L::tensor_type& gbmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    const typename L::tensor_type &y, typename L::size_type kl,
    typename L::size_type ku, const typename L::value_type &alpha,
    const typename L::value_type &beta) {
}

template < typename L >
const typename L::tensor_type& gemv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta) {
}

template < typename L >
const typename L::tensor_type& ger(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &a, const typename L::value_type &alpha) {
}

template < typename L >
const typename L::tensor_type& gerc(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &a, const typename L::value_type &alpha) {
}

template < typename L >
const typename L::tensor_type& hbmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    typename L::size_type k, const typename L::value_type &beta,
    typename L::BlasUplo uplo) {
}

template < typename L >
const typename L::tensor_type& hemv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::BlasUplo uplo) {
}

template < typename L >
const typename L::tensor_type& her(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &a,
    const typename L::value_type &alpha, typename L::BlasUplo uplo) {
}

template < typename L >
const typename L::tensor_type& her2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &a, const typename L::value_type &alpha,
    typename L::BlasUplo uplo) {
}

template < typename L >
const typename L::tensor_type& hpmv(
    L *l, const typename L::tensor_type &ap, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::BlasUplo uplo) {
}

template < typename L >
const typename L::tensor_type& hpr(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &ap,
    const typename L::value_type &alpha, typename L::BlasUplo uplo) {
}

template < typename L >
const typename L::tensor_type& hpr2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &ap, const typename L::value_type &alpha,
    typename L::BlasUplo uplo) {
}

template < typename L >
const typename L::tensor_type& sbmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    const typename L::tensor_type &y, typename L::size_type k,
    const typename L::value_type &alpha, const typename L::value_type &beta,
    typename L::BlasUplo uplo) {
}

template < typename L >
const typename L::tensor_type& spmv(
    L *l, const typename L::tensor_type &ap, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::BlasUplo) {
}

template < typename L >
const typename L::tensor_type& spr(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &ap,
    const typename L::value_type &alpha, typename L::BlasUplo uplo) {
}

template < typename L >
const typename L::tensor_type& spr2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &ap, const typename L::value_type &alpha,
    typename L::BlasUplo uplo) {
}

template < typename L >
const typename L::tensor_type& symv(
    L *l, const typename L::tensor_type &ap, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::BlasUplo uplo) {
}

template < typename L >
const typename L::tensor_type& syr(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &a,
    const typename L::value_type &alpha, typename L::BlasUplo uplo) {
}

template < typename L >
const typename L::tensor_type& syr2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &a, const typename L::value_type &alpha,
    typename L::BlasUplo uplo) {
}

template < typename L >
const typename L::tensor_type& tbmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::size_type k, typename L::BlasUplo uplo,
    typename L::BlasDiag diag) {
}

template < typename L >
const typename L::tensor_type& tbsv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::size_type k, typename L::BlasUplo uplo,
    typename L::BlasDiag diag) {
}

template < typename L >
const typename L::tensor_type& tpmv(
    L *l, const typename L::tensor_type &ap, const typename L::tensor_type &x,
    typename L::BlasUplo uplo, typename L::BlasDiag diag) {
}

template < typename L >
const typename L::tensor_type& tpsv(
    L *l, const typename L::tensor_type &ap, const typename L::tensor_type &x,
    typename L::BlasUplo uplo, typename L::BlasDiag diag) {
}

template < typename L >
const typename L::tensor_type& trmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::BlasUplo uplo, typename L::BlasDiag diag) {
}

template < typename L >
const typename L::tensor_type& trsv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::BlasUplo uplo, typename L::BlasDiag diag) {
}

}  // namespace math
}  // namespace linalg
}  // namespace thunder

#endif  // THUNDER_LINALG_MATH_INL_BLASLEVEL2_HPP_
