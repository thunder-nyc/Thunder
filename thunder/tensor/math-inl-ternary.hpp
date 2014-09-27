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

#ifndef THUNDER_TENSOR_MATH_INL_TERNARY_HPP_
#define THUNDER_TENSOR_MATH_INL_TERNARY_HPP_

#include "thunder/tensor/math.hpp"
#include "thunder/tensor/math-inl.hpp"

#include <cmath>
#include <complex>

#include "thunder/exception.hpp"

namespace thunder {
namespace tensor {
namespace math {

template < typename T >
const T& fma(
    const T &x, typename T::const_reference y, typename T::const_reference z) {
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::size_type x_length = x.length();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = static_cast< typename T::value_type >(
          ::std::fma(x_pointer[i * x_step], y, z));
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = static_cast< typename T::value_type >(::std::fma(*x_begin, y, z));
    }
  }
  return x;
}

template < typename T >
const T& fma(const T &x, const T &y, typename T::const_reference z) {
  if (x.length() != y.length()) {
    throw out_of_range("Tensors have different length");
  }
  if (x.partialContiguity(0, x.dimension() - 1) &&
      y.partialContiguity(0, y.dimension() - 1)) {
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::size_type x_length = x.length();
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_step = y.stride(y.dimension() - 1);
    for (typename T::size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = static_cast< typename T::value_type >(
          ::std::fma(x_pointer[i * x_step], y_pointer[i * y_step], z));
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(), y_begin = y.reference_begin();
         x_begin != x_end; ++x_begin, ++y_begin) {
      *x_begin = static_cast< typename T::value_type >(
          ::std::fma(*x_begin, *y_begin, z));
    }
  }
  return x;
}

template < typename T >
const T& fma(const T &x, typename T::const_reference y, const T &z) {
  if (x.length() != z.length()) {
    throw out_of_range("Tensors have different length");
  }
  if (x.partialContiguity(0, x.dimension() - 1) &&
      z.partialContiguity(0, z.dimension() - 1)) {
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::size_type x_length = x.length();
    typename T::pointer z_pointer = z.data();
    typename T::difference_type z_step = z.stride(z.dimension() - 1);
    for (typename T::size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = static_cast< typename T::value_type >(
          ::std::fma(x_pointer[i * x_step], y, z_pointer[i * z_step]));
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(), z_begin = z.reference_begin();
         x_begin != x_end; ++x_begin, ++z_begin) {
      *x_begin = static_cast< typename T::value_type >(
          ::std::fma(*x_begin, y, *z_begin));
    }
  }
  return x;
}

template < typename T >
const T& fma(const T &x, const T &y, const T &z) {
  if (x.length() != y.length() || x.length() != z.length()) {
    throw out_of_range("Tensors have different length");
  }
  if (x.partialContiguity(0, x.dimension() - 1) &&
      y.partialContiguity(0, y.dimension() - 1) &&
      z.partialContiguity(0, z.dimension() - 1)) {
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::size_type x_length = x.length();
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_step = y.stride(y.dimension() - 1);
    typename T::pointer z_pointer = z.data();
    typename T::difference_type z_step = z.stride(z.dimension() - 1);
    for (typename T::size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = static_cast< typename T::value_type >(::std::fma(
          x_pointer[i * x_step], y_pointer[i * y_step], z_pointer[i * z_step]));
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(), y_begin = y.reference_begin(),
             z_begin = z.reference_begin(); x_begin != x_end;
         ++x_begin, ++y_begin, ++z_begin) {
      *x_begin = static_cast< typename T::value_type >(
          ::std::fma(*x_begin, *y_begin, *z_begin));
    }
  }
  return x;
}

}  // namespace math
}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_MATH_INL_TERNARY_HPP_
