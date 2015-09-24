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

#ifndef THUNDER_TENSOR_COMPLEX_INL_REDUCTION_HPP_
#define THUNDER_TENSOR_COMPLEX_INL_REDUCTION_HPP_

#include "thunder/tensor/complex.hpp"
#include "thunder/tensor/complex-inl.hpp"

#include <cmath>
#include <complex>

#include "thunder/tensor/math.hpp"

namespace thunder {
namespace tensor {
namespace math {

template < typename D, typename A >
typename Tensor< Storage< ::std::complex< D >, A > >::value_type max(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    Tensor< typename Tensor< Storage< ::std::complex< D >, A > >
    ::size_storage> *pos) {
  throw domain_error("max is undefined for complex numbers.");
  return ::std::complex< D >(0, 0);
}

template < typename D, typename A >
typename Tensor< Storage< ::std::complex< D >, A > >::value_type min(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    Tensor< typename Tensor< Storage< ::std::complex< D >, A > >
    ::size_storage> *pos) {
  throw domain_error("max is undefined for complex numbers.");
  return ::std::complex< D >(0, 0);
}

template < typename D, typename A >
typename Tensor< Storage< ::std::complex< D >, A > >::value_type max(
    const Tensor< Storage< ::std::complex< D >, A > > &x) {
  throw domain_error("max is undefined for complex numbers.");
  return ::std::complex< D >(0, 0);
}

template < typename D, typename A >
typename Tensor< Storage< ::std::complex< D >, A > >::value_type min(
    const Tensor< Storage< ::std::complex< D >, A > > &x) {
  throw domain_error("min is undefined for complex numbers.");
  return ::std::complex< D >(0, 0);
}

template < typename D, typename A >
typename Tensor< Storage< ::std::complex< D >, A > >::value_type var(
    const Tensor< Storage< ::std::complex< D >, A > > &x) {
  typedef Tensor< Storage< ::std::complex< D >, A > > T;
  typename T::value_type sum_value = 0;
  typename T::value_type mean_value = x.mean();
  typename T::size_type x_length = x.length();
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::pointer x_pointer = x.data();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      sum_value += (x_pointer[i * x_step] - mean_value) *
          ::std::conj(x_pointer[i * x_step] - mean_value);
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      sum_value += (*x_begin - mean_value) * ::std::conj(*x_begin - mean_value);
    }
  }
  return sum_value / static_cast< typename T::value_type >(x_length);
}

template < typename D, typename A >
Tensor< Storage< ::std::complex< D >, A > > max(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::dim_type d,
    Tensor< typename Tensor< Storage< ::std::complex< D >, A > >
    ::size_storage> *pos) {
  throw domain_error("max is undefined for complex numbers.");
  return Tensor< Storage< ::std::complex< D >, A > >();
}

template < typename D, typename A >
Tensor< Storage< ::std::complex< D >, A > > min(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::dim_type d,
    Tensor< typename Tensor< Storage< ::std::complex< D >, A > >
    ::size_storage> *pos) {
  throw domain_error("max is undefined for complex numbers.");
  return Tensor< Storage< ::std::complex< D >, A > >();
}

template < typename D, typename A >
Tensor< Storage< ::std::complex< D >, A > > max(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::dim_type d) {
  throw domain_error("max is undefined for complex numbers.");
  return Tensor< Storage< ::std::complex< D >, A > >();
}

template < typename D, typename A >
Tensor< Storage< ::std::complex< D >, A > > min(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::dim_type d) {
  throw domain_error("min is undefined for complex numbers.");
  return Tensor< Storage< ::std::complex< D >, A > >();
}

template < typename D, typename A >
Tensor< Storage< ::std::complex< D >, A > > var(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::dim_type d) {
  typedef Tensor< Storage< ::std::complex< D >, A > > T;
  if (d >= x.dimension()) {
    throw out_of_range("Dimension exceeds limit.");
  }
  T t = x.mean(d);
  if (x.partialContiguity(0, d > 0 ? (d - 1) : 0)
      && x.partialContiguity(d + 1, x.dimension() - 1)) {
    // Get data pointers
    typename T::pointer x_data = x.data();
    typename T::pointer t_data = t.data();
    // Determine left and right size
    typename T::size_type x_left_length = 1;
    for (typename T::dim_type i = 0; i < d; ++i) {
      x_left_length *= x.size(i);
    }
    typename T::size_type x_length = x.size(d);
    typename T::size_type x_right_length = 1;
    for (typename T::dim_type i = d + 1; i < x.dimension(); ++i) {
      x_right_length *= x.size(i);
    }
    // Determine left and right step
    typename T::difference_type x_left_step = d > 0 ? x.stride(d - 1) : 0;
    typename T::difference_type x_step = x.stride(d);
    typename T::difference_type x_right_step = x.stride(x.dimension() - 1);
    typename T::difference_type t_left_step = d > 0 ? t.stride(d - 1) : 0;
    typename T::difference_type t_right_step = t.stride(t.dimension() - 1);
    // Run the loop
    typename T::value_type sum_value = 0;
    typename T::value_type mean_value = 0;
    typename T::value_type current_value = 0;
    for (typename T::size_type i = 0; i < x_left_length; ++i) {
      for (typename T::size_type j = 0; j < x_right_length; ++j) {
        sum_value = 0;
        mean_value = t_data[i * t_left_step + j * t_right_step];
        for (typename T::size_type k = 0; k < x_length; ++k) {
          current_value =
              x_data[i * x_left_step + j * x_right_step + k * x_step];
          sum_value += (current_value - mean_value) *
              ::std::conj(current_value - mean_value);
        }
        t_data[i * t_left_step + j * t_right_step] =
            sum_value / static_cast< typename T::value_type >(x_length);
      }
    }
  } else {
    typename T::size_storage position;
    typename T::size_type size_d = x.size(d);
    typename T::value_type sum_value = 0;
    typename T::value_type mean_value = 0;
    typename T::value_type current_value = 0;
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      sum_value = 0;
      mean_value = *t_begin;
      position = t_begin.position();
      for (typename T::size_type i = 0; i < size_d; ++i) {
        position[d] = i;
        current_value = x(position);
        sum_value += (current_value - mean_value) *
            ::std::conj(current_value - mean_value);
      }
      *t_begin = sum_value / static_cast< typename T::value_type >(size_d);
    }
  }
  return t;
}

}  // namespace math
}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_COMPLEX_INL_REDUCTION_HPP_
