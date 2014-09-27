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

#ifndef THUNDER_TENSOR_MATH_INL_UNARY_HPP_
#define THUNDER_TENSOR_MATH_INL_UNARY_HPP_

#include "thunder/tensor/math.hpp"
#include "thunder/tensor/math-inl.hpp"

#include <cmath>
#include <complex>

namespace thunder {
namespace tensor {
namespace math {

#define THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(func)                      \
  template < typename T >                                               \
  const T& func(const T &x) {                                           \
    if (x.partialContiguity(0, x.dimension() - 1)) {                    \
      typename T::pointer x_pointer = x.data();                         \
      typename T::difference_type x_step = x.stride(x.dimension() - 1); \
      typename T::size_type x_length = x.length();                      \
      for (typename T::size_type i = 0; i < x_length; ++i) {            \
        x_pointer[i * x_step] = static_cast< typename T::value_type >(  \
            ::std::func(x_pointer[i * x_step]));                        \
      }                                                                 \
    } else {                                                            \
      for (typename T::reference_iterator x_begin = x.reference_begin(), \
               x_end = x.reference_end(); x_begin != x_end; ++x_begin) { \
        *x_begin = static_cast< typename T::value_type >(               \
            ::std::func(*x_begin));                                     \
      }                                                                 \
    }                                                                   \
    return x;                                                           \
  }

THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(fabs);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(exp);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(exp2);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(expm1);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(log);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(log10);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(log2);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(log1p);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(sqrt);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(cbrt);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(sin);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(cos);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(tan);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(asin);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(acos);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(atan);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(sinh);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(cosh);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(tanh);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(asinh);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(acosh);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(atanh);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(erf);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(erfc);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(tgamma);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(lgamma);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(ceil);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(floor);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(trunc);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(round);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(nearbyint);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(rint);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(logb);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(fpclassify);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(isfinite);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(isinf);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(isnan);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(isnormal);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(signbit);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(real);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(imag);
THUNDER_TENSOR_MATH_DEFINE_STD_UNARY(arg);

#undef THUNDER_TENSOR_MATH_DEFINE_STD_UNARY

template < typename T >
const T& abs(const T &x) {
  return fabs(x);
}

template < typename T >
const T& cnrm(const T &x) {
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::size_type x_length = x.length();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = static_cast< typename T::value_type >(
          ::std::norm(x_pointer[i * x_step]));
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = static_cast< typename T::value_type >(
          ::std::norm(*x_begin));
    }
  }
  return x;
}

template < typename T >
const T& zero(const T &x) {
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::size_type x_length = x.length();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      x_pointer[i * x_step] = 0;
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      *x_begin = 0;
    }
  }
  return x;
}

template < typename T >
const T& conj(const T &x) {
  return x;
}

template < typename T >
const T& proj(const T &x) {
  return x;
}

}  // namespace math
}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_MATH_INL_UNARY_HPP_
