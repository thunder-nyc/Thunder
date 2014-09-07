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

#ifndef THUNDER_TENSOR_TENSOR_INL_BINARY_HPP_
#define THUNDER_TENSOR_TENSOR_INL_BINARY_HPP_

#include "thunder/tensor/tensor.hpp"
#include "thunder/tensor/tensor-inl.hpp"

#include <cmath>
#include <complex>

#include "thunder/exception.hpp"

namespace thunder {
namespace tensor {

#define THUNDER_TENSOR_DEFINE_EXTRA_BINARY(func)                        \
  template < typename S >                                               \
  Tensor< S >& Tensor< S >::func(const_reference y) const {             \
    return const_cast< Tensor& >(                                       \
        const_cast< const Tensor* >(this)->func(y));                    \
  }                                                                     \
  template < typename S >                                               \
  Tensor< S > Tensor< S >:func(const Tensor &x, const_reference y) {    \
    return x.clone().func(y);                                           \
  }                                                                     \
  template < typename S >                                               \
  Tensor< S >& Tensor< S >::func(const Tensor &y) const {               \
    return const_cast< Tensor& >(                                       \
        const_cast< const Tensor* >(this)->func(y));                    \
  }                                                                     \
  template < typename S >                                               \
  Tensor< S > Tensor< S >:func(const Tensor &x, const Tensor &y) {      \
    return x.clone().func(y);                                           \
  }

template < typename S >
const Tensor< S >& Tensor< S >::add(const_reference y) const {
  if (patialContiguity(0, size_.size() - 1)) {
    pointer x_pointer = data();
    difference_type x_step = stride_[stride_.size() - 1];
    size_type x_length = length();
    for (size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = x_pointer[i * x_step] + y;
    }
  } else {
    for (reference_iterator x_begin = reference_begin(),
             x_end = reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = *x_begin + y;
    }
  }
  return *this;
}
template < typename S >
const Tensor< S >& Tensor< S >::add(const Tensor &y) const {
  if (length() != y.length()) {
    throw out_of_range("Input tensor does not have equal length");
  }
  if (patialContiguity(0, size_.size() - 1) &&
      y.partialContiguity(0, y.dimension() - 1)) {
    pointer x_pointer = data();
    difference_type x_step = stride_[stride_.size() - 1];
    size_type x_length = length();
    pointer y_pointer = y.data();
    difference_type y_step = y.stride(y.dimension() - 1);
    for (size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = x_pointer[i * x_step] + y_data[i * y_step];
    }
  } else {
    for (reference_iterator x_begin = reference_begin(),
             x_end = reference_end(), y_begin = y.reference_begin();
         x_begin != x_end; ++x_begin, ++y_begin) {
      *x_begin = *x_begin + *y_begin;
    }
  }
  return *this;
}
THUNDER_TENSOR_DEFINE_EXTRA_BINARY(add);

template < typename S >
const Tensor< S >& Tensor< S >::sub(const_reference y) const {
  if (patialContiguity(0, size_.size() - 1)) {
    pointer x_pointer = data();
    difference_type x_step = stride_[stride_.size() - 1];
    size_type x_length = length();
    for (size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = x_pointer[i * x_step] - y;
    }
  } else {
    for (reference_iterator x_begin = reference_begin(),
             x_end = reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = *x_begin - y;
    }
  }
  return *this;
}
template < typename S >
const Tensor< S >& Tensor< S >::sub(const Tensor &y) const {
  if (length() != y.length()) {
    throw out_of_range("Input tensor does not have equal length");
  }
  if (patialContiguity(0, size_.size() - 1) &&
      y.partialContiguity(0, y.dimension() - 1)) {
    pointer x_pointer = data();
    difference_type x_step = stride_[stride_.size() - 1];
    size_type x_length = length();
    pointer y_pointer = y.data();
    difference_type y_step = y.stride(y.dimension() - 1);
    for (size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = x_pointer[i * x_step] - y_data[i * y_step];
    }
  } else {
    for (reference_iterator x_begin = reference_begin(),
             x_end = reference_end(), y_begin = y.reference_begin();
         x_begin != x_end; ++x_begin, ++y_begin) {
      *x_begin = *x_begin - *y_begin;
    }
  }
  return *this;
}
THUNDER_TENSOR_DEFINE_EXTRA_BINARY(sub);

template < typename S >
const Tensor< S >& Tensor< S >::mul(const_reference y) const {
  if (patialContiguity(0, size_.size() - 1)) {
    pointer x_pointer = data();
    difference_type x_step = stride_[stride_.size() - 1];
    size_type x_length = length();
    for (size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = x_pointer[i * x_step] * y;
    }
  } else {
    for (reference_iterator x_begin = reference_begin(),
             x_end = reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = (*x_begin) * y;
    }
  }
  return *this;
}
template < typename S >
const Tensor< S >& Tensor< S >::mul(const Tensor &y) const {
  if (length() != y.length()) {
    throw out_of_range("Input tensor does not have equal length");
  }
  if (patialContiguity(0, size_.size() - 1) &&
      y.partialContiguity(0, y.dimension() - 1)) {
    pointer x_pointer = data();
    difference_type x_step = stride_[stride_.size() - 1];
    size_type x_length = length();
    pointer y_pointer = y.data();
    difference_type y_step = y.stride(y.dimension() - 1);
    for (size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = x_pointer[i * x_step] * y_data[i * y_step];
    }
  } else {
    for (reference_iterator x_begin = reference_begin(),
             x_end = reference_end(), y_begin = y.reference_begin();
         x_begin != x_end; ++x_begin, ++y_begin) {
      *x_begin = (*x_begin) * (*y_begin);
    }
  }
  return *this;
}
THUNDER_TENSOR_DEFINE_EXTRA_BINARY(mul);

template < typename S >
const Tensor< S >& Tensor< S >::div(const_reference y) const {
  if (patialContiguity(0, size_.size() - 1)) {
    pointer x_pointer = data();
    difference_type x_step = stride_[stride_.size() - 1];
    size_type x_length = length();
    for (size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = x_pointer[i * x_step] / y;
    }
  } else {
    for (reference_iterator x_begin = reference_begin(),
             x_end = reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = (*x_begin) / y;
    }
  }
  return *this;
}
template < typename S >
const Tensor< S >& Tensor< S >::div(const Tensor &y) const {
  if (length() != y.length()) {
    throw out_of_range("Input tensor does not have equal length");
  }
  if (patialContiguity(0, size_.size() - 1) &&
      y.partialContiguity(0, y.dimension() - 1)) {
    pointer x_pointer = data();
    difference_type x_step = stride_[stride_.size() - 1];
    size_type x_length = length();
    pointer y_pointer = y.data();
    difference_type y_step = y.stride(y.dimension() - 1);
    for (size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = x_pointer[i * x_step] / y_data[i * y_step];
    }
  } else {
    for (reference_iterator x_begin = reference_begin(),
             x_end = reference_end(), y_begin = y.reference_begin();
         x_begin != x_end; ++x_begin, ++y_begin) {
      *x_begin = (*x_begin) / (*y_begin);
    }
  }
  return *this;
}
THUNDER_TENSOR_DEFINE_EXTRA_BINARY(div);

#define THUNDER_TENSOR_DEFINE_STD_BINARY(func)                          \
  template < typename S >                                               \
  const Tensor< S >& Tensor< S >::func(const_reference y) const {       \
  if (patialContiguity(0, size_.size() - 1)) {                          \
    pointer x_pointer = data();                                         \
    difference_type x_step = stride_[stride_.size() - 1];               \
    size_type x_length = length();                                      \
    for (size_type i = 0; i < x_length; ++i) {                          \
      x_pointer[i * x_step] = static_cast< value_type >(                \
          ::std::func(x_pointer[i * x_step], y));                       \
    }                                                                   \
  } else {                                                              \
    for (reference_iterator x_begin = reference_begin(),                \
             x_end = reference_end(); x_begin != x_end; ++x_begin) {    \
      *x_begin = static_cast< value_type >(::std::func(*x_begin, y));   \
    }                                                                   \
  }                                                                     \
  return *this;                                                         \
  }                                                                     \
  template < typename S >                                               \
  const Tensor< S >& Tensor< S >::func(const Tensor &y) const {         \
    if (length() != y.length()) {                                       \
      throw out_of_range("Input tensor does not have equal length");    \
    }                                                                   \
    if (patialContiguity(0, size_.size() - 1) &&                        \
        y.partialContiguity(0, y.dimension() - 1)) {                    \
      pointer x_pointer = data();                                       \
      difference_type x_step = stride_[stride_.size() - 1];             \
      size_type x_length = length();                                    \
      pointer y_pointer = y.data();                                     \
      difference_type y_step = y.stride(y.dimension() - 1);             \
      for (size_type i = 0; i < x_length; ++i) {                        \
        x_pointer[i * x_step] = static_cast< value_type >(              \
            ::std::func(x_pointer[i * x_step], y_data[i * y_step]));    \
      }                                                                 \
    } else {                                                            \
      for (reference_iterator x_begin = reference_begin(),              \
               x_end = reference_end(), y_begin = y.reference_begin();  \
           x_begin != x_end; ++x_begin, ++y_begin) {                    \
        *x_begin = static_cast< value_type > (                          \
            ::std::func(*x_begin, *y_begin));                           \
      }                                                                 \
    }                                                                   \
    return *this;                                                       \
  }                                                                     \
  THUNDER_TENSOR_DEFINE_EXTRA_BINARY(func);

THUNDER_TENSOR_DEFINE_STD_BINARY(fmod);
THUNDER_TENSOR_DEFINE_STD_BINARY(remainder);
THUNDER_TENSOR_DEFINE_STD_BINARY(fmax);
THUNDER_TENSOR_DEFINE_STD_BINARY(fmin);
THUNDER_TENSOR_DEFINE_STD_BINARY(fdim);
THUNDER_TENSOR_DEFINE_STD_BINARY(pow);
THUNDER_TENSOR_DEFINE_STD_BINARY(hypot);
THUNDER_TENSOR_DEFINE_STD_BINARY(atan2);
THUNDER_TENSOR_DEFINE_STD_BINARY(scalbn);
THUNDER_TENSOR_DEFINE_STD_BINARY(scalbln);
THUNDER_TENSOR_DEFINE_STD_BINARY(nextafter);
THUNDER_TENSOR_DEFINE_STD_BINARY(nexttoward);
THUNDER_TENSOR_DEFINE_STD_BINARY(copysign);
THUNDER_TENSOR_DEFINE_STD_BINARY(isgreater);
THUNDER_TENSOR_DEFINE_STD_BINARY(isgreaterequal);
THUNDER_TENSOR_DEFINE_STD_BINARY(isless);
THUNDER_TENSOR_DEFINE_STD_BINARY(islessequal);
THUNDER_TENSOR_DEFINE_STD_BINARY(islessgreater);
THUNDER_TENSOR_DEFINE_STD_BINARY(isunordered);

#undef THUNDER_TENSOR_DEFINE_STD_BINARY;
#undef THUNDER_TENSOR_DEFINE_EXTRA_BINARY;

template < typename S >
const Tensor< S >& Tensor< S >::fill(const_reference y) const {
  if (patialContiguity(0, size_.size() - 1)) {
    pointer x_pointer = data();
    difference_type x_step = stride_[stride_.size() - 1];
    size_type x_length = length();
    for (size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = y;
    }
  } else {
    for (reference_iterator x_begin = reference_begin(),
             x_end = reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = y;
    }
  }
  return *this;
}
template < typename S >
Tensor< S >& Tensor< S >::add(const_reference y) {
  return const_cast< Tensor& >(const_cast< const Tensor* >(this)->fill(y));
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_BINARY_HPP_
