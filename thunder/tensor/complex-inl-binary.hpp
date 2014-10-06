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
THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(hypot);
THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(atan2);
THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(ldexp);
THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(scalbn);
THUNDER_TENSOR_COMPLEX_UNDEFINED_BINARY(scalbln);
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

}  // namespace math
}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_COMPLEX_INL_BINARY_HPP_
