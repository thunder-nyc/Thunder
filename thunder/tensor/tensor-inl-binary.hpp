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

#include "thunder/exception.hpp"
#include "thunder/tensor/math.hpp"

namespace thunder {
namespace tensor {

#define THUNDER_TENSOR_DEFINE_BINARY(func)                              \
  template < typename S >                                               \
  const Tensor< S >& Tensor< S >::func(const_reference y) const {       \
    return math::func(*this, y);                                        \
  }                                                                     \
  template < typename S >                                               \
  Tensor< S >& Tensor< S >::func(const_reference y) {                   \
    return const_cast< Tensor& >(                                       \
        const_cast< const Tensor* >(this)->func(y));                    \
  }                                                                     \
  template < typename S >                                               \
  Tensor< S > Tensor< S >::func(const Tensor &x, const_reference y) {   \
    return x.clone().func(y);                                           \
  }                                                                     \
  template < typename S >                                               \
  const Tensor< S >& Tensor< S >::func(const Tensor &y) const {         \
    return math::func(*this, y);                                        \
  }                                                                     \
  template < typename S >                                               \
  Tensor< S >& Tensor< S >::func(const Tensor &y) {                     \
    return const_cast< Tensor& >(                                       \
        const_cast< const Tensor* >(this)->func(y));                    \
  }                                                                     \
  template < typename S >                                               \
  Tensor< S > Tensor< S >::func(const Tensor &x, const Tensor &y) {     \
    return x.clone().func(y);                                           \
  }

THUNDER_TENSOR_DEFINE_BINARY(add);
THUNDER_TENSOR_DEFINE_BINARY(sub);
THUNDER_TENSOR_DEFINE_BINARY(mul);
THUNDER_TENSOR_DEFINE_BINARY(div);
THUNDER_TENSOR_DEFINE_BINARY(fmod);
THUNDER_TENSOR_DEFINE_BINARY(remainder);
THUNDER_TENSOR_DEFINE_BINARY(fmax);
THUNDER_TENSOR_DEFINE_BINARY(fmin);
THUNDER_TENSOR_DEFINE_BINARY(fdim);
THUNDER_TENSOR_DEFINE_BINARY(pow);
THUNDER_TENSOR_DEFINE_BINARY(hypot);
THUNDER_TENSOR_DEFINE_BINARY(atan2);
THUNDER_TENSOR_DEFINE_BINARY(ldexp);
THUNDER_TENSOR_DEFINE_BINARY(scalbn);
THUNDER_TENSOR_DEFINE_BINARY(scalbln);
THUNDER_TENSOR_DEFINE_BINARY(nextafter);
THUNDER_TENSOR_DEFINE_BINARY(nexttoward);
THUNDER_TENSOR_DEFINE_BINARY(copysign);
THUNDER_TENSOR_DEFINE_BINARY(isgreater);
THUNDER_TENSOR_DEFINE_BINARY(isgreaterequal);
THUNDER_TENSOR_DEFINE_BINARY(isless);
THUNDER_TENSOR_DEFINE_BINARY(islessequal);
THUNDER_TENSOR_DEFINE_BINARY(islessgreater);
THUNDER_TENSOR_DEFINE_BINARY(isunordered);

#undef THUNDER_TENSOR_DEFINE_BINARY

template < typename S >
const Tensor< S >& Tensor< S >::fill(const_reference y) const {
  return math::fill(*this, y);
}
template < typename S >
Tensor< S >& Tensor< S >::fill(const_reference y) {
  return const_cast< Tensor& >(const_cast< const Tensor* >(this)->fill(y));
}
template < typename S >
Tensor< S > Tensor< S >::fill(const Tensor &x, const_reference y) {
  return x.clone().fill(y);
}

template < typename S >
template < typename T >
const Tensor< S >& Tensor< S >::copy(const T &y) const {
  return math::copy(*this, y);
}
template < typename S >
template < typename T >
Tensor< S >& Tensor< S >::copy(const T &y) {
  return const_cast< Tensor& >(const_cast< const Tensor *>(this)->copy(y));
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_BINARY_HPP_
