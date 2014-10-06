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

#ifndef THUNDER_TENSOR_TENSOR_INL_TERNARY_HPP_
#define THUNDER_TENSOR_TENSOR_INL_TERNARY_HPP_

#include "thunder/tensor/tensor.hpp"
#include "thunder/tensor/tensor-inl.hpp"

#include "thunder/exception.hpp"
#include "thunder/tensor/math.hpp"
#include "thunder/tensor/complex.hpp"

namespace thunder {
namespace tensor {

template < typename S >
template < typename TR >
const Tensor< S >& Tensor< S >::polar(
    typename TR::const_reference r, const TR& theta) const {
  return math::polar(*this, r, theta);
}
template < typename S >
template < typename TR >
const Tensor< S >& Tensor< S >::polar(
    const TR& r, typename TR::const_reference theta) const {
  return math::polar(*this, r, theta);
}
template < typename S >
template < typename TR >
const Tensor< S >& Tensor< S >::polar(const TR& r, const TR& theta) const {
  return math::polar(*this, r, theta);
}

template < typename S >
template < typename TR >
Tensor< S >& Tensor< S >::polar(
    typename TR::const_reference r, const TR& theta) {
  return const_cast< Tensor& >(
      const_cast< const Tensor& >(this)->polar(r, theta));
}
template < typename S >
template < typename TR >
Tensor< S >& Tensor< S >::polar(
    const TR& r, typename TR::const_reference theta) {
  return const_cast< Tensor& >(
      const_cast< const Tensor& >(this)->polar(r, theta));
}
template < typename S >
template < typename TR >
Tensor< S >& Tensor< S >::polar(const TR& r, const TR& theta) {
  return const_cast< Tensor& >(
      const_cast< const Tensor& >(this)->polar(r, theta));
}

template < typename S >
template < typename TR >
Tensor< S > Tensor< S >::polar(
    const Tensor &x, typename TR::const_reference r, const TR& theta) {
  return x.clone().polar(r, theta);
}
template < typename S >
template < typename TR >
Tensor< S > Tensor< S >::polar(
    const Tensor &x, const TR& r, typename TR::const_reference theta) {
  return x.clone().polar(r, theta);
}
template < typename S >
template < typename TR >
Tensor< S > Tensor< S >::polar(
    const Tensor &x, const TR& r, const TR& theta) {
  return x.clone().polar(r, theta);
}

template < typename S >
const Tensor< S >& Tensor< S >::polar(
    const_reference y, const_reference z) const {
  return math::polar(*this, y, z);
}
template < typename S >
const Tensor< S >& Tensor< S >::polar(
    const Tensor &y, const_reference z) const {
  return math::polar(*this, y, z);
}
template < typename S >
const Tensor< S >& Tensor< S >::polar(
    const_reference y, const Tensor& z) const {
  return math::polar(*this, y, z);
}
template < typename S >
const Tensor< S >& Tensor< S >::polar(
    const Tensor &y, const Tensor &z) const {
  return math::polar(*this, y, z);
}

template < typename S >
Tensor< S >& Tensor< S >::polar(const_reference y, const_reference z) {
  return const_cast< Tensor& >(const_cast< const Tensor* >(this)->polar(y, z));
}
template < typename S >
Tensor< S >& Tensor< S >::polar(const Tensor &y, const_reference z) {
  return const_cast< Tensor& >(const_cast< const Tensor* >(this)->polar(y, z));
}
template < typename S >
Tensor< S >& Tensor< S >::polar(const_reference y, const Tensor &z) {
  return const_cast< Tensor& >(const_cast< const Tensor* >(this)->polar(y, z));
}
template < typename S >
Tensor< S >& Tensor< S >::polar(const Tensor &y, const Tensor &z) {
  return const_cast< Tensor& >(const_cast< const Tensor* >(this)->polar(y, z));
}

template < typename S >
Tensor< S > Tensor< S >::polar(
    const Tensor &x, const_reference y, const_reference z) {
  return x.clone().polar(y, z);
}
template < typename S >
Tensor< S > Tensor< S >::polar(
    const Tensor &x, const Tensor &y, const_reference z) {
  return x.clone().polar(y, z);
}
template < typename S >
Tensor< S > Tensor< S >::polar(
    const Tensor &x, const_reference y, const Tensor &z) {
  return x.clone().polar(y, z);
}
template < typename S >
Tensor< S > Tensor< S >::polar(
    const Tensor &x, const Tensor &y, const Tensor &z) {
  return x.clone().polar(y, z);
}

template < typename S >
const Tensor< S >& Tensor< S >::fma(
    const_reference y, const_reference z) const {
  return math::fma(*this, y, z);
}
template < typename S >
const Tensor< S >& Tensor< S >::fma(const Tensor &y, const_reference z) const {
  return math::fma(*this, y, z);
}
template < typename S >
const Tensor< S >& Tensor< S >::fma(const_reference y, const Tensor& z) const {
  return math::fma(*this, y, z);
}
template < typename S >
const Tensor< S >& Tensor< S >::fma(const Tensor &y, const Tensor &z) const {
  return math::fma(*this, y, z);
}

template < typename S >
Tensor< S >& Tensor< S >::fma(const_reference y, const_reference z) {
  return const_cast< Tensor& >(const_cast< const Tensor* >(this)->fma(y, z));
}
template < typename S >
Tensor< S >& Tensor< S >::fma(const Tensor &y, const_reference z) {
  return const_cast< Tensor& >(const_cast< const Tensor* >(this)->fma(y, z));
}
template < typename S >
Tensor< S >& Tensor< S >::fma(const_reference y, const Tensor &z) {
  return const_cast< Tensor& >(const_cast< const Tensor* >(this)->fma(y, z));
}
template < typename S >
Tensor< S >& Tensor< S >::fma(const Tensor &y, const Tensor &z) {
  return const_cast< Tensor& >(const_cast< const Tensor* >(this)->fma(y, z));
}

template < typename S >
Tensor< S > Tensor< S >::fma(
    const Tensor &x, const_reference y, const_reference z) {
  return x.clone().fma(y, z);
}
template < typename S >
Tensor< S > Tensor< S >::fma(
    const Tensor &x, const Tensor &y, const_reference z) {
  return x.clone().fma(y, z);
}
template < typename S >
Tensor< S > Tensor< S >::fma(
    const Tensor &x, const_reference y, const Tensor &z) {
  return x.clone().fma(y, z);
}
template < typename S >
Tensor< S > Tensor< S >::fma(
    const Tensor &x, const Tensor &y, const Tensor &z) {
  return x.clone().fma(y, z);
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_TERNARY_HPP_
