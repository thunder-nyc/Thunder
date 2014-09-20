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

#ifndef THUNDER_TENSOR_TENSOR_INL_OPERATOR_HPP_
#define THUNDER_TENSOR_TENSOR_INL_OPERATOR_HPP_

#include "thunder/tensor/tensor.hpp"
#include "thunder/tensor/tensor-inl.hpp"

namespace thunder {
namespace tensor {

template < typename S >
Tensor< S > Tensor< S >::operator+(const_reference value) const {
  return clone().add(value);
}

template < typename S >
Tensor< S > Tensor< S >::operator-(const_reference value) const {
  return clone().sub(value);
}

template < typename S >
Tensor< S > operator+(
    typename Tensor< S >::const_reference value, const Tensor< S > &x) {
  return x.clone().add(value);
}

template < typename S >
Tensor< S > operator-(
    typename Tensor< S >::const_reference value, const Tensor< S > &x) {
  return x.clone().sub(value).mul(-1);
}

template < typename S >
Tensor< S >& Tensor< S >::operator+=(const_reference value) const {
  return add(value);
}

template < typename S >
Tensor< S >& Tensor< S >::operator-=(const_reference value) const {
  return sub(value);
}

template < typename S >
Tensor< S > Tensor< S >::operator+(const Tensor &y) const {
  return clone().add(y);
}

template < typename S >
Tensor< S > Tensor< S >::operator-(const Tensor &y) const {
  return clone().sub(y);
}

template < typename S >
Tensor< S >& Tensor< S >::operator+=(const Tensor &y) const {
  return add(y);
}

template < typename S >
Tensor< S >& Tensor< S >::operator-=(const Tensor &y) const {
  return sub(y);
}

template < typename S >
Tensor< S > Tensor< S >::operator==(const_reference value) const {
  return clone().islessgreater(value).islessgreater(
      static_cast< value_type >(true));
}

template < typename S >
Tensor< S > Tensor< S >::operator!=(const_reference value) const {
  return clone().islessgreater(value);
}

template < typename S >
Tensor< S > Tensor< S >::operator>(const_reference value) const {
  return clone().isgreater(value);
}

template < typename S >
Tensor< S > Tensor< S >::operator<(const_reference value) const {
  return clone().isless(value);
}

template < typename S >
Tensor< S > Tensor< S >::operator>=(const_reference value) const {
  return clone().isgreaterequal(value);
}

template < typename S >
Tensor< S > Tensor< S >::operator<=(const_reference value) const {
  return clone().islessequal(value);
}

template < typename S >
Tensor< S > operator==(
    typename Tensor< S >::const_reference value, const Tensor< S > &x) {
  return x.clone().islessgreater(value).islessgreater(
      static_cast< typename Tensor< S >::value_type >(true));
}

template < typename S >
Tensor< S > operator!=(
    typename Tensor< S >::const_reference value, const Tensor< S > &x) {
  return x.clone().islessgreater(value);
}

template < typename S >
Tensor< S > operator>(
    typename Tensor< S >::const_reference value, const Tensor< S > &x) {
  return x.clone().isless(value);
}

template < typename S >
Tensor< S > operator<(
    typename Tensor< S >::const_reference value, const Tensor< S > &x) {
  return x.clone().isgreater(value);
}

template < typename S >
Tensor< S > operator>=(
    typename Tensor< S >::const_reference value, const Tensor< S > &x) {
  return x.clone().islessequal(value);
}

template < typename S >
Tensor< S > operator<=(
    typename Tensor< S >::const_reference value, const Tensor< S > &x) {
  return x.clone().isgreaterequal(value);
}

template < typename S >
Tensor< S > Tensor< S >::operator==(const Tensor &y) const {
  return clone().islessgreater(y).islessgreater(
      static_cast< value_type >(true));
}

template < typename S >
Tensor< S > Tensor< S >::operator!=(const Tensor &y) const {
  return clone().islessgreater(y);
}

template < typename S >
Tensor< S > Tensor< S >::operator>(const Tensor &y) const {
  return clone().isgreater(y);
}

template < typename S >
Tensor< S > Tensor< S >::operator<(const Tensor &y) const {
  return clone().isless(y);
}

template < typename S >
Tensor< S > Tensor< S >::operator>=(const Tensor &y) const {
  return clone().isgreaterequal(y);
}

template < typename S >
Tensor< S > Tensor< S >::operator<=(const Tensor &y) const {
  return clone().islessequal(y);
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_OPERATOR_HPP_
