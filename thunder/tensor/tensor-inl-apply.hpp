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

#ifndef THUNDER_TENSOR_TENSOR_INL_APPLY_HPP_
#define THUNDER_TENSOR_TENSOR_INL_APPLY_HPP_

#include "thunder/tensor/tensor.hpp"
#include "thunder/tensor/tensor-inl.hpp"

namespace thunder {
namespace tensor {

template< typename S >
const Tensor< S >& Tensor< S >::apply(
    const ::std::function< value_type(value_type) > &lambda) const {
  if (partialContiguity(0, size_.size() - 1)) {
    pointer data_pointer = data();
    size_type data_length = length();
    difference_type step = stride_[stride_.size() - 1];
    for (size_type i = 0; i < data_length; ++i) {
      data_pointer[i * step] = lambda(data_pointer[i * step]);
    }
  } else {
    for (reference_iterator begin = reference_begin(), end = reference_end();
         begin != end; ++begin) {
      *begin = lambda(*begin);
    }
  }
  return *this;
}

template< typename S >
const Tensor< S >& Tensor< S >:: apply(
    const ::std::function< value_type(const value_type&) > &lambda) const {
  if (partialContiguity(0, size_.size() - 1)) {
    pointer data_pointer = data();
    size_type data_length = length();
    difference_type step = stride_[stride_.size() - 1];
    for (size_type i = 0; i < data_length; ++i) {
      data_pointer[i * step] = lambda(data_pointer[i * step]);
    }
  } else {
    for (reference_iterator begin = reference_begin(), end = reference_end();
         begin != end; ++begin) {
      *begin = lambda(*begin);
    }
  }
  return *this;
}

template< typename S >
const Tensor< S >& Tensor< S >::apply(
    const ::std::function< void(value_type&) > &lambda) const {
  if (partialContiguity(0, size_.size() - 1)) {
    pointer data_pointer = data();
    size_type data_length = length();
    difference_type step = stride_[stride_.size() - 1];
    for (size_type i = 0; i < data_length; ++i) {
      data_pointer[i * step] = lambda(data_pointer[i * step]);
    }
  } else {
    for (reference_iterator begin = reference_begin(), end = reference_end();
         begin != end; ++begin) {
      *begin = lambda(*begin);
    }
  }
  return *this;
}

template< typename S >
const Tensor< S >& Tensor< S >::apply(
    const ::std::function< void(value_type*) > &lambda) const;


template< typename S >
Tensor< S >& Tensor< S >::apply(
    const ::std::function< value_type(value_type) > &lambda);

template< typename S >
Tensor< S >& Tensor< S >::apply(
    const ::std::function< value_type(const value_type&) > &lambda);

template< typename S >
Tensor< S >& Tensor< S >::apply(
    const ::std::function< void(value_type&) > &lambda);

template< typename S >
Tensor< S >& Tensor< S >::apply(
    const ::std::function< void(value_type*) > &lambda);

// Static lambda applications are delegated
template< typename S >
Tensor< S > Tensor< S >::apply(
    const Tensor& x, const ::std::function< value_type(value_type) > &lambda);

template< typename S >
Tensor< S > Tensor< S >::apply(
    const Tensor &x,
    const ::std::function< value_type(const value_type&) > &lambda);

template< typename S >
Tensor< S > Tensor< S >::apply(
    const Tensor &x,
    const ::std::function< void(value_type&) > &lambda);

template< typename S >
Tensor< S > Tensor< S >::apply(
    const Tensor &x,
    const ::std::function< void(value_type*) > &lambda);

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_APPLY_HPP_
