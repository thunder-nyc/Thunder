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

#ifndef THUNDER_TENSOR_MATH_INL_COMPLEX_HPP_
#define THUNDER_TENSOR_MATH_INL_COMPLEX_HPP_

#include "thunder/tensor/math.hpp"
#include "thunder/tensor/math-inl.hpp"

#include <cmath>
#include <complex>

#include "thunder/storage.hpp"

namespace thunder {
namespace tensor {
namespace math {

#define THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(func)               \
  template < typename D, typename A >                                   \
  const Tensor< Storage< ::std::complex< D >, A > >& func(              \
      const Tensor< Storage< ::std::complex< D >, A > > &x) {           \
    throw domain_error(#func " is undefined for complex numbers");      \
    return x;                                                           \
  }

THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(fabs);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(exp2);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(expm1);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(log1p);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(cbrt);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(erf);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(erfc);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(tgamma);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(lgamma);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(ceil);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(floor);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(trunc);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(round);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(nearbyint);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(rint);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(logb);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(fpclassify);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(isfinite);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(isinf);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(log2);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(isnan);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(isnormal);
THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX(signbit);

#undef THUNDER_TENSOR_MATH_UNDEFINED_UNARY_COMPLEX

#define THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(func)              \
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

THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(fmod);
THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(remainder);
THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(fmax);
THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(fmin);
THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(fdim);
THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(hypot);
THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(atan2);
THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(ldexp);
THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(scalbn);
THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(scalbln);
THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(nextafter);
THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(nexttoward);
THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(copysign);
THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(isgreater);
THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(isgreaterequal);
THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(isless);
THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(islessequal);
THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(islessgreater);
THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX(isunordered);

#undef THUNDER_TENSOR_MATH_UNDEFINED_BINARY_COMPLEX

#define THUNDER_TENSOR_MATH_UNDEFINED_REDUCTION_COMPLEX(func)           \
  template < typename D, typename A >                                   \
  typename Tensor< Storage< ::std::complex< D >, A > >::value_type      \
  func(const Tensor< Storage< ::std::complex< D >, A > > &x) {          \
    throw domain_error(#func " is undefined for complex numbers");      \
    return 0;                                                           \
  }                                                                     \
  template < typename D, typename A >                                   \
  Tensor< Storage< ::std::complex< D >, A > >                           \
  func(const Tensor< Storage< ::std::complex< D >, A > > &x,            \
       typename Tensor< Storage< ::std::complex< D >, A > >::dim_type d) { \
    throw domain_error(#func " is undefined for complex numbers");      \
    return Tensor< Storage< ::std::complex< D >, A > >();               \
  }

THUNDER_TENSOR_MATH_UNDEFINED_REDUCTION_COMPLEX(max);
THUNDER_TENSOR_MATH_UNDEFINED_REDUCTION_COMPLEX(min);

#undef THUNDER_TENSOR_MATH_UNDEFINED_REDUCTION_COMPLEX

template < typename D, typename A >
typename Tensor< Storage< ::std::complex< D >, A > >::value_type
max(const Tensor< Storage< ::std::complex< D >, A > > &x,
    Tensor< typename Tensor< Storage< ::std::complex< D >, A > >
    ::size_storage> *pos) {
  throw domain_error("max is undefined for complex numbers");
  return 0;
}

template < typename D, typename A >
typename Tensor< Storage< ::std::complex< D >, A > >::value_type
min(const Tensor< Storage< ::std::complex< D >, A > > &x,
    Tensor< typename Tensor< Storage< ::std::complex< D >, A > >
    ::size_storage> *pos) {
  throw domain_error("max is undefined for complex numbers");
  return 0;
}

template < typename D, typename A >
Tensor< Storage< ::std::complex< D >, A > > max(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::dim_type d,
    Tensor< typename Tensor< Storage< ::std::complex< D >, A > >
    ::size_storage> *pos) {
  throw domain_error("max is undefined for complex numbers");
  return Tensor< Storage< ::std::complex< D >, A > >();
}

template < typename D, typename A >
Tensor< Storage< ::std::complex< D >, A > > min(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::dim_type d,
    Tensor< typename Tensor< Storage< ::std::complex< D >, A > >
    ::size_storage> *pos) {
  throw domain_error("max is undefined for complex numbers");
  return Tensor< Storage< ::std::complex< D >, A > >();
}

template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& fma(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference z) {
  throw domain_error("fma is undefined for complex numbers");
  return x;
}

template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& fma(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference z) {
  throw domain_error("fma is undefined for complex numbers");
  return x;
}

template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& fma(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    typename Tensor< Storage< ::std::complex< D >, A > >::const_reference y,
    const Tensor< Storage< ::std::complex< D >, A > > &z) {
  return x;
}

template < typename D, typename A >
const Tensor< Storage< ::std::complex< D >, A > >& fma(
    const Tensor< Storage< ::std::complex< D >, A > > &x,
    const Tensor< Storage< ::std::complex< D >, A > > &y,
    const Tensor< Storage< ::std::complex< D >, A > > &z) {
  return x;
}

}  // namespace math
}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_MATH_INL_COMPLEX_HPP_
