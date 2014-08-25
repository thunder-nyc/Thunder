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

#ifndef THUNDER_TENSOR_TENSOR_INL_TYPE_HPP_
#define THUNDER_TENSOR_TENSOR_INL_TYPE_HPP_

#include "thunder/tensor/tensor.hpp"
#include "thunder/tensor/tensor-inl.hpp"

namespace thunder {
namespace tensor {

template < typename S >
template < typename T >
T Tensor< S >::type() {
  typename T::size_storage sz(size_.size());
  typename T::stride_storage st(stride_.size());
  for (dim_type i = 0; i < size_.size(); ++i) {
    sz[i] = static_cast< typename T::size_type >(size_[i]);
    st[i] = static_cast< typename T::difference_type >(stride_[i]);
  }
  return T(sz, st).copy(*this);
}

template < typename S >
const Tensor< S >& Tensor< S >::type() const {
  return *this;
}

template < typename S >
Tensor< S >& Tensor< S >::type() {
  return *this;
}

// Static type conversions are delegated
template < typename S >
template < typename T >
T Tensor< S >::type(const Tensor& x) {
  return x.type< T >();
}

template < typename S >
Tensor< S >& Tensor< S >::type(Tensor &x) {
  return x.type< Tensor >();
}

template < typename S >
const Tensor< S >& Tensor< S >::type(const Tensor &x) {
  return x.type< const Tensor >();
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_TYPE_HPP_
