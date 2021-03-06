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

#ifndef THUNDER_TENSOR_TENSOR_INL_ACCESS_HPP_
#define THUNDER_TENSOR_TENSOR_INL_ACCESS_HPP_

#include "thunder/tensor/tensor.hpp"
#include "thunder/tensor/tensor-inl.hpp"

#include <memory>
#include <utility>

#include "thunder/exception.hpp"

namespace thunder {
namespace tensor {

template < typename S >
Tensor< S >& Tensor< S >::operator=(Tensor y) {
  std::swap(size_, y.size_);
  std::swap(stride_, y.stride_);
  std::swap(storage_, y.storage_);
  std::swap(offset_, y.offset_);
  return *this;
}

template < typename S >
typename Tensor< S >::reference Tensor< S >::operator()() const {
  return (*storage_)[offset_];
}

template < typename S >
typename Tensor< S >::reference Tensor< S >::operator()(size_type pos) const {
  return (*storage_)[offset_ + pos * stride_[0]];
}

template < typename S >
typename Tensor< S >::reference Tensor< S >::operator()(
    size_type pos0, size_type pos1) const {
  return (*storage_)[offset_ + pos0 * stride_[0] + pos1 * stride_[1]];
}

template < typename S >
typename Tensor< S >::reference Tensor< S >::operator()(
    size_type pos0, size_type pos1, size_type pos2) const {
  return (*storage_)[offset_ + pos0 * stride_[0] + pos1 * stride_[1]
                     + pos2 * stride_[2]];
}


template < typename S >
typename Tensor< S >::reference Tensor< S >::operator()(
    size_type pos0, size_type pos1, size_type pos2, size_type pos3) const {
  return (*storage_)[offset_ + pos0 * stride_[0] + pos1 * stride_[1]
                   + pos2 * stride_[2] + pos3 * stride_[3]];
}

template < typename S >
typename Tensor< S >::reference Tensor< S >::operator()(
    const size_storage &pos) const {
  size_type os = offset_;
  for (dim_type i = 0; i < pos.size(); ++i) {
    os += pos[i] * stride_[i];
  }
  return (*storage_)[os];
}

template < typename S >
Tensor< S > Tensor< S >::operator[](size_type pos) const {
  size_type os = offset_ + pos * stride_[0];
  if (size_.size() > 1) {
    size_storage sz(size_.size()-1);
    stride_storage st(stride_.size()-1);
    for (dim_type i = 0; i < sz.size(); ++i) {
      sz[i] = size_[i + 1];
      st[i] = stride_[i + 1];
    }
    return Tensor(sz, st, storage_, os);
  } else {
    size_storage sz(1, 1);
    stride_storage st(1, 1);
    return Tensor(sz, st, storage_, os);
  }
}

template < typename S >
Tensor< S > Tensor< S >::operator[](const size_storage &pos) const {
  size_type os = offset_;
  for (dim_type i = 0; i < pos.size(); ++i) {
    os += pos[i] * stride_[i];
  }
  if (size_.size() > pos.size()) {
    size_storage sz(size_.size() - pos.size());
    stride_storage st(stride_.size() - pos.size());
    for (dim_type i = 0; i < sz.size(); ++i) {
      sz[i] = size_[pos.size() + i];
      st[i] = stride_[pos.size() + i];
    }
    return Tensor(sz, st, storage_, os);
  } else {
    size_storage sz(1, 1);
    stride_storage st(1, 1);
    return Tensor(sz, st, storage_, os);
  }
}

template < typename S >
Tensor< S > Tensor< S >::operator[](
    const Storage< ::std::pair< size_type, size_type > > &range) const {
  size_type os = offset_;
  size_storage sz(size_);
  for (dim_type i = 0; i < range.size(); ++i) {
    os = os + range[i].first * stride_[i];
    sz[i] = range[i].second - range[i].first + 1;
  }
  return Tensor(sz, stride_, storage_, os);
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_ACCESS_HPP_
