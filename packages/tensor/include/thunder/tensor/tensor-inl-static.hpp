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
Tensor< S > Tensor< S >::ones(size_type n, allocator_type alloc) {
  return Tensor(n, alloc).fill(1);
}

template < typename S >
Tensor< S > Tensor< S >::ones(size_type m, size_type n, allocator_type alloc) {
  return Tensor(m, n, alloc).fill(1);
}

template < typename S >
Tensor< S > Tensor< S >::ones(size_type n0, size_type n1, size_type n2,
                              allocator_type alloc) {
  return Tensor(n0, n1, n2, alloc).fill(1);
}

template < typename S >
Tensor< S > Tensor< S >::ones(
    size_type n0, size_type n1, size_type n2, size_type n3,
    allocator_type alloc) {
  return Tensor(n0, n1, n2, n3, alloc).fill(1);
}

template < typename S >
Tensor< S > Tensor< S >::ones(const size_storage &sz, allocator_type alloc) {
  return Tensor(sz, alloc).fill(1);
}

template < typename S >
Tensor< S > Tensor< S >::zeros(size_type n, allocator_type alloc) {
  return Tensor(n, alloc).zero();
}

template < typename S >
Tensor< S > Tensor< S >::zeros(size_type m, size_type n, allocator_type alloc) {
  return Tensor(m, n, alloc).zero();
}

template < typename S >
Tensor< S > Tensor< S >::zeros(size_type n0, size_type n1, size_type n2,
                               allocator_type alloc) {
  return Tensor(n0, n1, n2, alloc).zero();
}

template < typename S >
Tensor< S > Tensor< S >::zeros(
    size_type n0, size_type n1, size_type n2, size_type n3,
    allocator_type alloc) {
  return Tensor(n0, n1, n2, n3, alloc).zero();
}

template < typename S >
Tensor< S > Tensor< S >::zeros(const size_storage &sz, allocator_type alloc) {
  return Tensor(sz, alloc).zero();
}

template < typename S >
template < typename TR >
Tensor< S > Tensor< S >::polars(
    typename TR::const_reference r, const TR& theta, allocator_type alloc) {
  Tensor x(alloc);
  x.resizeAs(theta);
  return x.polar(r, theta);
}

template < typename S >
template < typename TR >
Tensor< S > Tensor< S >::polars(
    const TR& r, typename TR::const_reference theta, allocator_type alloc) {
  Tensor x(alloc);
  x.resizeAs(r);
  return x.polar(r, theta);
}

template < typename S >
template < typename TR >
Tensor< S > Tensor< S >::polars(const TR& r, const TR& theta,
                                allocator_type alloc) {
  if (r.length() != theta.length()) {
    throw out_of_range("Tensors have different length.");
  }
  Tensor x(alloc);
  x.resizeAs(r);
  return x.polar(r, theta);
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_STATIC_HPP_
