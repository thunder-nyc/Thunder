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

#ifndef THUNDER_TENSOR_MATH_INL_BINARY_HPP_
#define THUNDER_TENSOR_MATH_INL_BINARY_HPP_

#include "thunder/tensor/math.hpp"
#include "thunder/tensor/math-inl.hpp"

#include <cmath>
#include <complex>

#include "thunder/exception.hpp"

namespace thunder {
namespace tensor {
namespace math {

template < typename T >
const T& add(const T &x, typename T::const_reference y) {
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::size_type x_length = x.length();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = x_pointer[i * x_step] + y;
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = (*x_begin) + y;
    }
  }
  return x;
}
template < typename T >
const T& add(const T &x, const T &y) {
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
      x_pointer[i * x_step] = x_pointer[i * x_step] + y_pointer[i * y_step];
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(), y_begin = y.reference_begin();
         x_begin != x_end; ++x_begin, ++y_begin) {
      *x_begin = (*x_begin) + (*y_begin);
    }
  }
  return x;
}

template < typename T >
const T& sub(const T &x, typename T::const_reference y) {
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::size_type x_length = x.length();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = x_pointer[i * x_step] - y;
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = (*x_begin) - y;
    }
  }
  return x;
}
template < typename T >
const T& sub(const T &x, const T &y) {
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
      x_pointer[i * x_step] = x_pointer[i * x_step] - y_pointer[i * y_step];
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(), y_begin = y.reference_begin();
         x_begin != x_end; ++x_begin, ++y_begin) {
      *x_begin = (*x_begin) - (*y_begin);
    }
  }
  return x;
}

template < typename T >
const T& mul(const T &x, typename T::const_reference y) {
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::size_type x_length = x.length();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = x_pointer[i * x_step] * y;
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = (*x_begin) * y;
    }
  }
  return x;
}
template < typename T >
const T& mul(const T &x, const T &y) {
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
      x_pointer[i * x_step] = x_pointer[i * x_step] * y_pointer[i * y_step];
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(), y_begin = y.reference_begin();
         x_begin != x_end; ++x_begin, ++y_begin) {
      *x_begin = (*x_begin) * (*y_begin);
    }
  }
  return x;
}

template < typename T >
const T& div(const T &x, typename T::const_reference y) {
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::size_type x_length = x.length();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = x_pointer[i * x_step] / y;
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = (*x_begin) / y;
    }
  }
  return x;
}
template < typename T >
const T& div(const T &x, const T &y) {
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
      x_pointer[i * x_step] = x_pointer[i * x_step] / y_pointer[i * y_step];
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(), y_begin = y.reference_begin();
         x_begin != x_end; ++x_begin, ++y_begin) {
      *x_begin = (*x_begin) / (*y_begin);
    }
  }
  return x;
}

#define THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(func)                     \
  template < typename T >                                               \
  const T& func(const T &x, typename T::const_reference y) {            \
    if (x.partialContiguity(0, x.dimension() - 1)) {                    \
      typename T::pointer x_pointer = x.data();                         \
      typename T::difference_type x_step = x.stride(x.dimension() - 1); \
      typename T::size_type x_length = x.length();                      \
      for (typename T::size_type i = 0; i < x_length; ++i) {            \
        x_pointer[i * x_step] = static_cast< typename T::value_type >(  \
            ::std::func(x_pointer[i * x_step], y));                     \
      }                                                                 \
    } else {                                                            \
      for (typename T::reference_iterator x_begin = x.reference_begin(), \
               x_end = x.reference_end(); x_begin != x_end; ++x_begin) { \
        *x_begin = static_cast< typename T::value_type >(               \
            ::std::func(*x_begin, y));                                  \
      }                                                                 \
    }                                                                   \
    return x;                                                           \
  }                                                                     \
  template < typename T >                                               \
  const T& func(const T &x, const T &y) {                               \
    if (x.length() != y.length()) {                                     \
      throw out_of_range("Tensors have different length");              \
    }                                                                   \
    if (x.partialContiguity(0, x.dimension() - 1) &&                    \
        y.partialContiguity(0, y.dimension() - 1)) {                    \
      typename T::pointer x_pointer = x.data();                         \
      typename T::difference_type x_step = x.stride(x.dimension() - 1); \
      typename T::size_type x_length = x.length();                      \
      typename T::pointer y_pointer = y.data();                         \
      typename T::difference_type y_step = y.stride(y.dimension() - 1); \
      for (typename T::size_type i = 0; i < x_length; ++i) {            \
        x_pointer[i * x_step] = static_cast< typename T::value_type >(  \
            ::std::func(x_pointer[i * x_step], y_pointer[i * y_step])); \
      }                                                                 \
    } else {                                                            \
      for (typename T::reference_iterator x_begin = x.reference_begin(), \
               x_end = x.reference_end(), y_begin = y.reference_begin(); \
           x_begin != x_end; ++x_begin, ++y_begin) {                    \
        *x_begin = static_cast< typename T::value_type > (              \
            ::std::func(*x_begin, *y_begin));                           \
      }                                                                 \
    }                                                                   \
    return x;                                                           \
  }

THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(fmod);
THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(remainder);
THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(fmax);
THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(fmin);
THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(fdim);
THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(pow);
THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(hypot);
THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(atan2);
THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(ldexp);
THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(scalbn);
THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(scalbln);
THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(nextafter);
THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(nexttoward);
THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(copysign);
THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(isgreater);
THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(isgreaterequal);
THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(isless);
THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(islessequal);
THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(islessgreater);
THUNDER_TENSOR_MATH_DEFINE_STD_BINARY(isunordered);

#undef THUNDER_TENSOR_MATH_DEFINE_STD_BINARY

template < typename T >
const T& fill(const T &x, typename T::const_reference y) {
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::size_type x_length = x.length();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = y;
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = y;
    }
  }
  return x;
}

}  // namespace math
}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_MATH_INL_BINARY_HPP_
