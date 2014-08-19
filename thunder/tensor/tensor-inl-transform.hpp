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

#ifndef THUNDER_TENSOR_TENSOR_INL_TRANSFORM_HPP_
#define THUNDER_TENSOR_TENSOR_INL_TRANSFORM_HPP_

#include "thunder/tensor/tensor.hpp"
#include "thunder/tensor/tensor-inl.hpp"

#include <utility>

#include "thunder/exception.hpp"

namespace thunder {
namespace tensor {

template < typename S >
template < typename T >
Tensor< S > Tensor< S >::viewAs(const T &y, size_type os = 0) const {
  size_storage sz(y.dimension());
  stride_storage st(y.dimension());
  for (dim_type i = 0; i < sz.size(); ++i) {
    sz[i] = static_cast< size_type >(y.size(i));
    st[i] = static_cast< difference_type >(y.stride(i));
  }
  return this->view(sz, st, os);
}

template < typename S >
template < typename T >
Tensor< S > viewAs(const T &y, const stride_storage &st,
                   size_type os = 0) const {
  size_storage sz(y.dimension());
  for (dim_type i = 0; i < sz.size(); ++i) {
    sz[i] = static_cast< size_type >(y.size(i));
  }
  return this->view(sz, st, os);
}

template < typename S >
template < typename T >
Tensor< S > Tensor< S >::extract(const T &y) const {
  if (dimension() < y.dimension()) {
    throw out_of_range("Dimension exceeds limit");
  }
  dim_type y_dimension = static_cast< dim_type > y.dimension();
  for (i = 0; i < y_dimension; ++i) {
    if (size_[i] != static_cast< size_type >(y.size(i))) {
      throw out_of_range("Size does not match");
    }
  }
  size_storage sz(size_.size() - y_dimension + 1);
  for (dim_type i = y_dimension; i < size_.size(); ++i) {
    sz[i - y_dimension + 1] = size_[i];
  }
  if (isContiguous() && y.isContiguous()) {
    y::size_type y_length = y.length();
    y::pointer y_data = y.data();
    for (y::size_type i = 0; i < y_length; ++y) {
      if (static_cast< bool >(y_data[i]) == true) {
        ++sz[0];
      }
    }
    Tensor< S > t(sz);
    difference_type st = stride_[y_dimension - 1];
    pointer t_data = t.data();
    pointer dt = data();
    size_type current;
    for (y::size_type i = 0; i < y_length; ++y) {
      if (static_cast< bool >(y_data[i]) == true) {
        for (size_type j = 0; j < st; ++j) {
          t_data[current++] = dt[i * st + j];
        }
      }
    }
    return t;
  } else {
    // TODO: Implement here. Need to access iterators.
  }
}

template < typename T >
Tensor shuffle(const T &y) const;


// Static templated subtensor extractors are delegated
template < typename T >
static Tensor viewAs(const Tensor &x, const T &y, size_type os = 0);
template < typename T >
static Tensor viewAs(const Tensor &x, const T &y,
                     const stride_storage &st, size_type os = 0);
template < typename T >
static Tensor extract(const Tensor &x, const T &y);
template < typename T >
static Tensor shuffle(const Tensor &x, const T &y);

}  // namespace tensor
}  // namespace thunder

#endif  //THUNDER_TENSOR_TENSOR_INL_TRANSFORM_HPP_
