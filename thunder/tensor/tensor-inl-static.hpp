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

#ifndef THUNDER_TENSOR_TENSOR_INL_STATIC_HPP_
#define THUNDER_TENSOR_TENSOR_INL_STATIC_HPP_

#include "thunder/tensor/tensor.hpp"
#include "thunder/tensor/tensor-inl.hpp"

#include "thunder/tensor/math.hpp"

namespace thunder {
namespace tensor {

template < typename S >
Tensor< S > Tensor< S >::ones(size_type n) {
  return Tensor(n).fill(1);
}

template < typename S >
Tensor< S > Tensor< S >::ones(size_type m, size_type n) {
  return Tensor(m, n).fill(1);
}

template < typename S >
Tensor< S > Tensor< S >::ones(size_type n0, size_type n1, size_type n2) {
  return Tensor(n0, n1, n2).fill(1);
}

template < typename S >
Tensor< S > Tensor< S >::ones(
    size_type n0, size_type n1, size_type n2, size_type n3) {
  return Tensor(n0, n1, n2, n3).fill(1);
}

template < typename S >
Tensor< S > Tensor< S >::ones(const size_storage &sz) {
  return Tensor(sz).fill(1);
}

template < typename S >
Tensor< S > Tensor< S >::zeros(size_type n) {
  return Tensor(n).zero();
}

template < typename S >
Tensor< S > Tensor< S >::zeros(size_type m, size_type n) {
  return Tensor(m, n).zero();
}

template < typename S >
Tensor< S > Tensor< S >::zeros(size_type n0, size_type n1, size_type n2) {
  return Tensor(n0, n1, n2).zero();
}

template < typename S >
Tensor< S > Tensor< S >::zeros(
    size_type n0, size_type n1, size_type n2, size_type n3) {
  return Tensor(n0, n1, n2, n3).zero();
}

template < typename S >
Tensor< S > Tensor< S >::zeros(const size_storage &sz) {
  return Tensor(sz).zero();
}

template < typename S >
template < typename TR >
Tensor< S > Tensor< S >::polar(
    typename TR::const_reference r, const TR& theta) {
  return math::polar< Tensor >(r, theta);
}

template < typename S >
template < typename TR >
Tensor< S > Tensor< S >::polar(
    const TR& r, typename TR::const_reference theta) {
  return math::polar< Tensor >(r, theta);
}

template < typename S >
template < typename TR >
Tensor< S > Tensor< S >::polar(const TR& r, const TR& theta) {
  return math::polar< Tensor >(r, theta);
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_STATIC_HPP_
