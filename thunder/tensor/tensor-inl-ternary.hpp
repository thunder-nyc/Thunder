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

#include <cmath>

#include "thunder/exception.hpp"

namespace thunder {
namespace tensor {

template < typename S >
const Tensor< S >& Tensor< S >::fma(
    const_reference y, const_reference z) const {
  if (partialContiguity(0, size_.size() - 1)) {
    pointer x_pointer = data();
    difference_type x_step = stride_[stride_.size() - 1];
    size_type x_length = length();
    for (size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = static_cast< value_type >(
          ::std::fma(x_pointer[i * x_step], y, z));
    }
  } else {
    for (reference_iterator x_begin = reference_begin(),
             x_end = reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = static_cast< value_type >(::std::fma(*x_begin, y, z));
    }
  }
}

template < typename S >
const Tensor< S >& Tensor< S >::fma(const Tensor &y, const_reference z) const {
  if (length() != y.length()) {
    throw out_of_range("Input tensor does not have equal length");
  }
  if (partialContiguity(0, size_.size() - 1) &&
      y.partialContiguity(0, y.dimension() - 1)) {
    pointer x_pointer = data();
    difference_type x_step = stride_[stride_.size() - 1];
    size_type x_length = length();
    pointer y_pointer = y.data();
    difference_type y_step = y.stride(y.dimension() - 1);
    for (size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = static_cast< value_type >(
          ::std::fma(x_pointer[i * x_step], y_pointer[i * y_step], z));
    }
  } else {
    for (reference_iterator x_begin = reference_begin(),
             x_end = reference_end(), y_begin = y.reference_begin();
         x_begin != x_end; ++x_begin, ++y_begin) {
      *x_begin = static_cast< value_type >(::std::fma(*x_begin, *y_begin, z));
    }
  }
}

template < typename S >
const Tensor< S >& Tensor< S >::fma(const_reference y, const Tensor& z) const {
  if (length() != z.length()) {
    throw out_of_range("Input tensor does not have equal length");
  }
  if (partialContiguity(0, size_.size() - 1) &&
      z.partialContiguity(0, z.dimension() - 1)) {
    pointer x_pointer = data();
    difference_type x_step = stride_[stride_.size() - 1];
    size_type x_length = length();
    pointer z_pointer = z.data();
    difference_type z_step = z.stride(z.dimension() - 1);
    for (size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = static_cast< value_type >(
          ::std::fma(x_pointer[i * x_step], y, z_pointer[i * z_step]));
    }
  } else {
    for (reference_iterator x_begin = reference_begin(),
             x_end = reference_end(), z_begin = z.reference_begin();
         x_begin != x_end; ++x_begin, ++z_begin) {
      *x_begin = static_cast< value_type >(::std::fma(*x_begin, y, *z_begin));
    }
  }
}

template < typename S >
const Tensor< S >& Tensor< S >::fma(const Tensor &y, const Tensor &z) const {
  if (length() != y.length() || length() != z.length()) {
    throw out_of_range("Input tensors do not have equal length");
  }
  if (partialContiguity(0, size_.size() - 1) &&
      y.partialContiguity(0, y.dimension() - 1) &&
      z.partialContiguity(0, z.dimension() - 1) {
    pointer x_pointer = data();
    difference_type x_step = stride_[stride_.size() - 1];
    size_type x_length = length();
    pointer y_pointer = y.data();
    difference_type y_step = y.stride(y.dimension() - 1);
    pointer z_pointer = z.data();
    difference_type z_step = z.stride(z.dimension() - 1);
    for (size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = static_cast< value_type >(::std::fma(
          x_pointer[i * x_step], y_pointer[i * y_step], z_pointer[i * z_step));
    }
  } else {
    for (reference_iterator x_begin = reference_begin(),
             x_end = reference_end(), y_begin = y.reference_begin(),
             z_begin = z.reference_begin(); x_begin != x_end;
         ++x_begin, ++y_begin, ++z_begin) {
      *x_begin = static_cast< value_type >(
          ::std::fma(*x_begin, *y_begin, *z_begin));
    }
  }
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
Tensor< S >& Tensor< S >::fma(
    const Tensor &x, const_reference y, const_reference z) {
  return x.clone().fma(y, z);
}
template < typename S >
Tensor< S >& Tensor< S >::fma(
    const Tensor &x, const Tensor &y, const_reference z) {
  return x.clone().fma(y, z);
}
template < typename S >
Tensor< S >& Tensor< S >::fma(
    const Tensor &x, const_reference y, const Tensor &z) {
  return x.clone().fma(y, z);
}
template < typename S >
Tensor< S >& Tensor< S >::fma(
    const Tensor &x, const Tensor &y, const Tensor &z) {
  return x.clone().fma(y, z);
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_TERNARY_HPP_
