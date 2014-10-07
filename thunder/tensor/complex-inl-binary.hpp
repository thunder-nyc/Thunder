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

#ifndef THUNDER_TENSOR_COMPLEX_INL_BINARY_HPP_
#define THUNDER_TENSOR_COMPLEX_INL_BINARY_HPP_

#include "thunder/tensor/complex.hpp"
#include "thunder/tensor/complex-inl.hpp"

#include <cmath>
#include <complex>
#include <limits>

#include "thunder/tensor/math.hpp"

namespace thunder {
namespace tensor {
namespace math {

#define THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(func)                   \
  template < typename D, typename A >                                   \
  const Tensor< Storage< ::std::complex< D >, A > >& func(              \
      const Tensor< Storage< ::std::complex< D >, A > > &x,             \
      typename Tensor< Storage< ::std::complex< D >, A > >::            \
      const_reference y) {                                              \
    throw domain_error(#func " is undefined for complex numbers");      \
    return x;                                                           \
  }                                                                     \
  template < typename D, typename A >                                   \
  const Tensor< Storage< ::std::complex< D >, A > >& func(              \
      const Tensor< Storage< ::std::complex< D >, A > > &x,             \
      const Tensor< Storage< ::std::complex< D >, A > > &y) {           \
    throw domain_error(#func " is undefined for complex numbers");      \
    return x;                                                           \
  }

THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(fmod);
THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(remainder);
THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(fmax);
THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(fmin);
THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(fdim);
THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(nextafter);
THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(nexttoward);
THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(copysign);
THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(isgreater);
THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(isgreaterequal);
THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(isless);
THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(islessequal);
THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(islessgreater);
THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(isunordered);

#undef THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY

template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& hypot(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y) {
  typedef Tensor< Storage< ::std::complex< D >, A > > T;
  D y_norm = ::std::norm(y);
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::size_type x_length = x.length();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] =
          ::std::hypot(::std::norm(x_pointer[i * x_step]), y_norm);
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = ::std::hypot(::std::norm(*x_begin), y_norm);
    }
  }
  return x;
}
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& hypot(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y) {
  typedef Tensor< Storage< ::std::complex< D >, A > > T;
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
      x_pointer[i * x_step] = ::std::hypot(::std::norm(x_pointer[i * x_step]),
                                           ::std::norm(y_pointer[i * y_step]));
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(), y_begin = y.reference_begin();
         x_begin != x_end; ++x_begin, ++y_begin) {
      *x_begin = ::std::hypot(::std::norm(*x_begin), ::std::norm(*y_begin));
    }
  }
  return x;
}

template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& atan2(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y) {
  typedef Tensor< Storage< ::std::complex< D >, A > > T;
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::size_type x_length = x.length();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = ::std::atan(x_pointer[i * x_step] / y);
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = ::std::atan((*x_begin) / y);
    }
  }
  return x;
}
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& atan2(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y) {
  typedef Tensor< Storage< ::std::complex< D >, A > > T;
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
      x_pointer[i * x_step] =
          ::std::atan(x_pointer[i * x_step] / y_pointer[i * y_step]);
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(), y_begin = y.reference_begin();
         x_begin != x_end; ++x_begin, ++y_begin) {
      *x_begin = ::std::atan((*x_begin) / (*y_begin));
    }
  }
  return x;
}

template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& ldexp(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y) {
  typedef Tensor< Storage< ::std::complex< D >, A > > T;
  ::std::complex< D > y_exp = ::std::pow(2, y);
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::size_type x_length = x.length();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = x_pointer[i * x_step] * y_exp;
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = (*x_begin) * y_exp;
    }
  }
  return x;
}
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& ldexp(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y) {
  typedef Tensor< Storage< ::std::complex< D >, A > > T;
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
      x_pointer[i * x_step] =
          x_pointer[i * x_step] * ::std::pow(2, y_pointer[i * y_step]);
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(), y_begin = y.reference_begin();
         x_begin != x_end; ++x_begin, ++y_begin) {
      *x_begin = (*x_begin) * ::std::pow(2, *y_begin);
    }
  }
  return x;
}

template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& scalbn(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y) {
  typedef Tensor< Storage< ::std::complex< D >, A > > T;
  ::std::complex< D > y_exp = ::std::pow(::std::numeric_limits< D >::radix, y);
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::size_type x_length = x.length();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = x_pointer[i * x_step] * y_exp;
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = (*x_begin) * y_exp;
    }
  }
  return x;
}
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& scalbn(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y) {
  typedef Tensor< Storage< ::std::complex< D >, A > > T;
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
      x_pointer[i * x_step] = x_pointer[i * x_step] *
          ::std::pow(::std::numeric_limits< D >::radix, y_pointer[i * y_step]);
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(), y_begin = y.reference_begin();
         x_begin != x_end; ++x_begin, ++y_begin) {
      *x_begin = (*x_begin) *
          ::std::pow(::std::numeric_limits< D >::radix, *y_begin);
    }
  }
  return x;
}

template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& scalbln(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y) {
  return scalbn(x, y);
}
template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& scalbln(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y) {
  return scalbn(x, y);
}

template< typename D, typename A, typename T1 >
const T1& copy(const T1 &x,
               const Tensor< Storage< ::std::complex< D >, A > > &y) {
  typedef Tensor< Storage< ::std::complex< D >, A > > T2;
  if (x.length() != y.length()) {
    throw out_of_range("Tensors have different length");
  }
  if (x.isContiguous() && y.isContiguous()) {
    typename T1::size_type x_length = x.length();
    typename T1::pointer x_data = x.data();
    typename T2::pointer y_data = y.data();
    for (typename T1::size_type i = 0; i < x_length; ++i) {
      x_data[i] = static_cast< typename T1::value_type >(
          ::std::real(y_data[i]));
    }
  } else {
    typename T2::reference_iterator y_begin = y.reference_begin();
    for (typename T1::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end();
         x_begin != x_end; ++x_begin, ++y_begin) {
      *x_begin = static_cast< typename T1::value_type>(
          ::std::real(*y_begin));
    }
  }
  return x;
}

template< typename D1, typename D2, typename A1, typename A2 >
const Tensor< Storage< ::std::complex< D1 >, A1 > >& copy(
    const Tensor< Storage< ::std::complex< D1 >, A1 > > &x,
    const Tensor< Storage< ::std::complex< D2 >, A2 > > &y) {
  typedef Tensor< Storage< ::std::complex< D1 >, A1 > > T1;
  typedef Tensor< Storage< ::std::complex< D2 >, A2 > > T2;
  if (x.length() != y.length()) {
    throw out_of_range("Tensors have different length");
  }
  if (x.isContiguous() && y.isContiguous()) {
    typename T1::size_type x_length = x.length();
    typename T1::pointer x_data = x.data();
    typename T2::pointer y_data = y.data();
    for (typename T1::size_type i = 0; i < x_length; ++i) {
      x_data[i] = typename T1::value_type(
          static_cast< D1 >(::std::real(y_data[i])),
          static_cast< D1 >(::std::imag(y_data[i])));
    }
  } else {
    typename T2::reference_iterator y_begin = y.reference_begin();
    for (typename T1::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end();
         x_begin != x_end; ++x_begin, ++y_begin) {
      *x_begin = typename T1::value_type(
          static_cast< D1 >(::std::real(*y_begin)),
          static_cast< D1 >(::std::imag(*y_begin)));
    }
  }
  return x;
}

}  // namespace math
}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_COMPLEX_INL_BINARY_HPP_
