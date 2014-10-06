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

#ifndef THUNDER_TENSOR_TENSOR_INL_REDUCTION_HPP_
#define THUNDER_TENSOR_TENSOR_INL_REDUCTION_HPP_

#include "thunder/tensor/tensor.hpp"
#include "thunder/tensor/tensor-inl.hpp"

#include <limits>
#include <cmath>

#include "thunder/exception.hpp"
#include "thunder/tensor/index_iterator.hpp"
#include "thunder/tensor/math.hpp"
#include "thunder/tensor/complex.hpp"

namespace thunder {
namespace tensor {

template < typename S >
typename Tensor< S >::value_type Tensor< S >::max(
    Tensor< size_storage > *pos) const {
  return math::max(*this, pos);
}
template < typename S >
typename Tensor< S >::value_type Tensor< S >::max(
    const Tensor &x, Tensor< size_storage > *pos) {
  return x.max(pos);
}
template < typename S >
Tensor< S > Tensor< S >::max(dim_type d, Tensor< size_storage > *pos) const {
  return math::max(*this, d, pos);
}
template < typename S >
Tensor< S > Tensor< S >::max(
    const Tensor &x, dim_type d, Tensor< size_storage > *pos) {
  return x.max(d, pos);
}

template < typename S >
typename Tensor< S >::value_type Tensor< S >::min(
    Tensor< size_storage > *pos) const {
  return math::min(*this, pos);
}
template < typename S >
typename Tensor< S >::value_type Tensor< S >::min(
    const Tensor &x, Tensor< size_storage > *pos) {
  return x.min(pos);
}
template < typename S >
Tensor< S > Tensor< S >::min(dim_type d, Tensor< size_storage > *pos) const {
  return math::min(*this, d, pos);
}
template < typename S >
Tensor< S > Tensor< S >::min(
    const Tensor &x, dim_type d, Tensor< size_storage > *pos) {
  return x.min(d, pos);
}

#define THUNDER_TENSOR_DEFINE_REDUCTION(func)                           \
  template < typename S >                                               \
  typename Tensor< S >::value_type Tensor< S >::func() const {          \
    return math::func(*this);                                           \
  }                                                                     \
  template < typename S >                                               \
  typename Tensor< S >::value_type Tensor< S >::func(const Tensor &x) { \
    return x.func();                                                    \
  }                                                                     \
  template < typename S >                                               \
  Tensor< S > Tensor< S >::func(dim_type d) const {                     \
    return math::func(*this, d);                                        \
  }                                                                     \
  template < typename S >                                               \
  Tensor< S > Tensor< S >::func(const Tensor &x, dim_type d) {          \
    return x.func(d);                                                   \
  }

THUNDER_TENSOR_DEFINE_REDUCTION(max);
THUNDER_TENSOR_DEFINE_REDUCTION(min);
THUNDER_TENSOR_DEFINE_REDUCTION(sum);
THUNDER_TENSOR_DEFINE_REDUCTION(prod);
THUNDER_TENSOR_DEFINE_REDUCTION(mean);
THUNDER_TENSOR_DEFINE_REDUCTION(var);
THUNDER_TENSOR_DEFINE_REDUCTION(std);

#undef THUNDER_TENSOR_DEFINE_REDUCTION

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_REDUCTION_HPP_
