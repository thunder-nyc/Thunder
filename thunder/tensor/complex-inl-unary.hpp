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

#ifndef THUNDER_TENSOR_COMPLEX_INL_UNARY_HPP_
#define THUNDER_TENSOR_COMPLEX_INL_UNARY_HPP_

#include "thunder/tensor/complex.hpp"
#include "thunder/tensor/complex-inl.hpp"

#include <cmath>
#include <complex>
#include <limits>

#include "thunder/tensor/math.hpp"

namespace thunder {
namespace tensor {
namespace math {

#define THUNDER_TENSOR_COMPLEX_UNDEFINED_UNARY(func)                    \
  template < typename D, typename A >                                   \
  const Tensor< Storage< ::std::complex< D >, A > >& func(              \
      const Tensor< Storage< ::std::complex< D >, A > > &x) {           \
    throw domain_error(#func " is undefined for complex numbers.");      \
    return x;                                                           \
  }

THUNDER_TENSOR_COMPLEX_UNDEFINED_UNARY(erf);
THUNDER_TENSOR_COMPLEX_UNDEFINED_UNARY(erfc);
THUNDER_TENSOR_COMPLEX_UNDEFINED_UNARY(tgamma);
THUNDER_TENSOR_COMPLEX_UNDEFINED_UNARY(lgamma);
THUNDER_TENSOR_COMPLEX_UNDEFINED_UNARY(ceil);
THUNDER_TENSOR_COMPLEX_UNDEFINED_UNARY(floor);
THUNDER_TENSOR_COMPLEX_UNDEFINED_UNARY(trunc);
THUNDER_TENSOR_COMPLEX_UNDEFINED_UNARY(round);
THUNDER_TENSOR_COMPLEX_UNDEFINED_UNARY(nearbyint);
THUNDER_TENSOR_COMPLEX_UNDEFINED_UNARY(rint);
THUNDER_TENSOR_COMPLEX_UNDEFINED_UNARY(fpclassify);
THUNDER_TENSOR_COMPLEX_UNDEFINED_UNARY(isfinite);
THUNDER_TENSOR_COMPLEX_UNDEFINED_UNARY(isinf);
THUNDER_TENSOR_COMPLEX_UNDEFINED_UNARY(isnan);
THUNDER_TENSOR_COMPLEX_UNDEFINED_UNARY(isnormal);
THUNDER_TENSOR_COMPLEX_UNDEFINED_UNARY(signbit);

#undef THUNDER_TENSOR_COMPLEX_UNDEFINED_UNARY

template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& fabs(
    const Tensor< Storage < ::std::complex< D >, A > > &x) {
  return abs(x);
}

template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& exp2(
    const Tensor< Storage< ::std::complex< D >, A > > &x) {
  typedef Tensor< Storage< ::std::complex< D >, A > > T;
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::size_type x_length = x.length();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = ::std::pow(2, x_pointer[i * x_step]);
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = ::std::pow(2, *x_begin);
    }
  }
  return x;
}

template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& expm1(
    const Tensor< Storage< ::std::complex< D >, A > > &x) {
  return sub(exp(x), 1);
}

template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& log1p(
    const Tensor< Storage< ::std::complex< D >, A > > &x) {
  return log(add(x, 1));
}

template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& cbrt(
    const Tensor< Storage< ::std::complex< D >, A > > &x) {
  return pow(x, 1/3);
}

template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& log2(
    const Tensor< Storage< ::std::complex< D >, A > > &x) {
  return div(log(x), ::std::log(2));
}

template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& logb(
    const Tensor< Storage< ::std::complex< D >, A > > &x) {
  return div(log(abs(x)), ::std::log(::std::numeric_limits< D >::radix));
}

template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& conj(
    const Tensor< Storage < ::std::complex< D >, A > > &x) {
  typedef Tensor< Storage< ::std::complex< D >, A > > T;
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::size_type x_length = x.length();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = ::std::conj(x_pointer[i * x_step]);
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = ::std::conj(*x_begin);
    }
  }
  return x;
}

template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& proj(
    const Tensor< Storage < ::std::complex< D >, A > > &x) {
  typedef Tensor< Storage< ::std::complex< D >, A > > T;
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::size_type x_length = x.length();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = ::std::proj(x_pointer[i * x_step]);
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = ::std::proj(*x_begin);
    }
  }
  return x;
}

}  // namespace math
}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_COMPLEX_INL_UNARY_HPP_
