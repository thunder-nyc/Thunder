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

#ifndef THUNDER_TENSOR_MATH_INL_STATIC_HPP_
#define THUNDER_TENSOR_MATH_INL_STATIC_HPP_

#include "thunder/tensor/math.hpp"
#include "thunder/tensor/math-inl.hpp"

#include <cmath>
#include <complex>

#include "thunder/exception.hpp"
#include "thunder/storage.hpp"
#include "thunder/tensor/tensor.hpp"

namespace thunder {
namespace tensor {
namespace math {

#define THUNDER_MATH_COMPLEX_TENSOR(D, A)       \
  Tensor< Storage< ::std::complex< D >, A > >

template < typename T1, typename T2 >
T1 polars(typename T2::const_reference r, const T2 &theta) {
  throw domain_error("polar is undefined for real numbers");
  return T1();
}
template < typename T1, typename T2 >
T1 polars(const T2 &r, typename T2::const_reference theta) {
  throw domain_error("polar is undefined for real numbers");
  return T1();
}
template < typename T1, typename T2 >
T1 polars(const T2 &r, const T2 &theta) {
  throw domain_error("polar is undefined for real numbers");
  return T1();
}

template < typename D, typename A, typename T2 >
THUNDER_MATH_COMPLEX_TENSOR(D, A) polars(
    typename T2::const_reference r, const T2 &theta) {
  typedef typename THUNDER_MATH_COMPLEX_TENSOR(D, A) T1;
  T1 t;
  t.resizeAs(theta);
  if (theta.partialContiguity(0, theta.dimension() - 1)) {
    typename T1::pointer t_data = t.data();
    typename T2::pointer theta_data = theta.data();
    typename T2::size_type theta_length = theta.length();
    typename T2::difference_type theta_step =
        theta.stride(theta.dimension() - 1);
    for (typename T2::size_type i = 0; i < theta_length; ++i) {
      t_data[i] = ::std::polar(r, theta_data[i * theta_step]);
    }
  } else {
    typename T1::pointer t_data = t.data();
    typename T1::size_type t_current = 0;
    for (typename T2::reference_iterator theta_begin = theta.reference_begin(),
             theta_end = theta.reference_end();
         theta_begin != theta_end; ++theta_begin) {
      t_data[t_current++] = ::std::polar(r, *theta_begin);
    }
  }
  return t;
}
template < typename D, typename A, typename T2 >
THUNDER_MATH_COMPLEX_TENSOR(D, A) polars(
    const T2 &r, typename T2::cont_reference theta) {
  typedef typename THUNDER_MATH_COMPLEX_TENSOR(D, A) T1;
  T1 t;
  t.resizeAs(r);
  if (r.partialContiguity(0, r.dimension() - 1)) {
    typename T1::pointer t_data = t.data();
    typename T2::pointer r_data = r.data();
    typename T2::size_type r_length = r.length();
    typename T2::difference_type r_step = r.stride(r.dimension() - 1);
    for (typename T2::size_type i = 0; i < r_length; ++i) {
      t_data[i] = ::std::polar(r_data[i * r_step], theta);
    }
  } else {
    typename T1::pointer t_data = t.data();
    typename T1::size_type t_current = 0;
    for (typename T2::reference_iterator r_begin = r.reference_begin(),
             r_end = r.reference_end(); r_begin != r_end; ++r_begin) {
      t_data[t_current++] = ::std::polar(*r_begin, theta);
    }
  }
  return t;
}
template < typename D, typename A, typename T2 >
THUNDER_MATH_COMPLEX_TENSOR(D, A) polars(const T2 &r, const T2 &theta) {
  if (r.length() != theta.length()) {
    throw out_of_range("Tensors have difference length");
  }
  typedef typename THUNDER_MATH_COMPLEX_TENSOR(D, A) T1;
  T1 t;
  t.resizeAs(r);
  if (r.partialContiguity(0, r.dimension() - 1) &&
      theta.partialContiguity(0, theta.dimension() - 1)) {
    typename T1::pointer t_data = t.data();
    typename T2::pointer r_data = r.data();
    typename T2::size_type r_length = r.length();
    typename T2::difference_type r_step = r.stride(r.dimension() - 1);
    typename T2::pointer theta_data = theta.data();
    typename T2::difference_type theta_step =
        theta.stride(theta.dimension() - 1);
    for (typename T2::size_type i = 0; i < r_length; ++i) {
      t_data[i] = ::std::polar(r_data[i * r_step], theta_data[i * theta_step]);
    }
  } else {
    typename T1::pointer t_data = t.data();
    typename T1::size_type t_current = 0;
    for (typename T2::reference_iterator r_begin = r.reference_begin(),
             r_end = r.reference_end, theta_begin = theta.reference_begin();
         r_begin != r_end; ++r_begin, ++theta_begin) {
      t_data[t_current++] = ::std::polar(*r_begin, *theta_begin);
    }
  }
}

#undef THUNDER_MATH_COMPLEX_TENSOR

}  // namespace math
}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_MATH_INL_STATIC_HPP_
