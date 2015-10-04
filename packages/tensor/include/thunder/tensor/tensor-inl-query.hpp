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

#ifndef THUNDER_TENSOR_TENSOR_INL_QUERY_HPP_
#define THUNDER_TENSOR_TENSOR_INL_QUERY_HPP_

#include "thunder/tensor/tensor.hpp"
#include "thunder/tensor/tensor-inl.hpp"

#include "thunder/exception.hpp"

namespace thunder {
namespace tensor {

template < typename S >
template < typename T >
bool Tensor< S >::isSameSizeAs(const T &y) const {
  if (size_.size() != y.dimension()) {
    return false;
  }
  for (dim_type i = 0; i < size_.size(); ++i) {
    if (size_[i] != y.size(i)) {
      return false;
    }
  }
  return true;
}

template < typename S >
template < typename T >
bool Tensor< S >::isSameSizeAs(const Tensor &x, const T &y) {
  return x.isSameSizeAs(y);
}

template < typename S >
typename Tensor< S >::dim_type Tensor< S >::dimension() const {
  return size_.size();
}

template < typename S >
typename Tensor< S >::size_storage Tensor< S >::size() const {
  return size_;
}

template < typename S >
typename Tensor< S >::size_type Tensor< S >::size(dim_type dim) const {
  if (dim >= size_.size()) {
    throw invalid_argument("Argument exceeds tensor dimension.");
  }
  return size_[dim];
}

template < typename S >
typename Tensor< S >::size_type Tensor< S >::length() const {
  size_type length = 1;
  for (dim_type i = 0; i < size_.size(); ++i) {
    length *= size_[i];
  }
  return length;
}

template < typename S >
typename Tensor< S >::stride_storage Tensor< S >::stride() const {
  return stride_;
}

template < typename S >
typename Tensor< S >::difference_type Tensor< S >::stride(dim_type dim) const {
  if (dim >= stride_.size()) {
    throw invalid_argument("Argument exceeds tensor dimension.");
  }
  return stride_[dim];
}

template < typename S >
typename Tensor< S >::storage_pointer Tensor< S >::storage() const {
  return storage_;
}

template < typename S >
typename Tensor< S >::size_type Tensor< S >::offset() const {
  return offset_;
}

template < typename S >
typename Tensor< S >::pointer Tensor< S >::data() const {
  return storage_->data() + offset_;
}

template < typename S >
typename Tensor< S >::allocator_type Tensor< S >::allocator() const {
  return storage_->allocator();
}

template < typename S >
typename Tensor< S >::reference Tensor< S >::get() const {
  return (*this)();
}

template < typename S >
typename Tensor< S >::reference Tensor< S >::get(size_type pos) const {
  if (pos >= size_[0]) {
    throw out_of_range("Position exceeds size limit.");
  }
  return (*this)(pos);
}

template < typename S >
typename Tensor< S >::reference Tensor< S >::get(
    size_type pos0, size_type pos1) const {
  if (size_.size() < 2 || pos0 >= size_[0] || pos1 >= size_[1]) {
    throw out_of_range("Position exceeds size limit.");
  }
  return (*this)(pos0, pos1);
}

template < typename S >
typename Tensor< S >::reference Tensor< S >::get(
    size_type pos0, size_type pos1, size_type pos2) const {
  if (size_.size() < 3 || pos0 >= size_[0] || pos1 >= size_[1]
      || pos2 >= size_[2]) {
    throw out_of_range("Position exceeds size limit.");
  }
  return (*this)(pos0, pos1, pos2);
}

template < typename S >
typename Tensor< S >::reference Tensor< S >::get(
    size_type pos0, size_type pos1, size_type pos2, size_type pos3) const {
  if (size_.size() < 4 || pos0 >= size_[0] || pos1 >= size_[1]
      || pos2 >= size_[2] || pos3 >= size_[3]) {
    throw out_of_range("Position exceeds size limit.");
  }
  return (*this)(pos0, pos1, pos2, pos3);
}
template < typename S >
typename Tensor< S >::reference Tensor< S >::get(
    const size_storage &pos) const {
  if (size_.size() < pos.size()) {
    throw out_of_range("Position exceeds size limit.");
  }
  for (dim_type i = 0; i < pos.size(); ++i) {
    if (pos[i] >= size_[i]) {
      throw out_of_range("Position exceeds size limit.");
    }
  }
  return (*this)(pos);
}

template < typename S >
bool Tensor< S >::isContiguous() const {
  if (stride_[stride_.size() - 1] != 1) {
    return false;
  }
  for (dim_type i = stride_.size() - 1; i > 0; --i) {
    if (stride_[i - 1] !=
        stride_[i] * static_cast< difference_type >(size_[i])) {
      return false;
    }
  }
  return true;
}

template < typename S >
bool Tensor< S >::partialContiguity(dim_type a, dim_type b) const {
  if (b < size_.size()) {
    for (dim_type i = a; i < b; ++i) {
      if (stride_[i] !=
          stride_[i + 1] * static_cast< difference_type >(size_[i + 1])) {
        return false;
      }
    }
  } else {
    if (stride_[stride_.size() - 1] != 1) {
      return false;
    }
    for (dim_type i = a; i < size_.size() - 1; ++i) {
      if (stride_[i] !=
          stride_[i + 1] * static_cast< difference_type >(size_[i + 1])) {
        return false;
      }
    }
  }
  return true;
}

template < typename S >
bool Tensor< S >::isUnique() const {
  return storage_.unique();
}

template < typename S >
typename Tensor< S >::dim_type Tensor< S >::dimension(const Tensor &x) {
  return x.dimension();
}

template < typename S >
typename Tensor< S >::size_storage Tensor< S >::size(const Tensor &x) {
  return x.size();
}

template < typename S >
typename Tensor< S >::size_type Tensor< S >::size(
    const Tensor &x, dim_type dim) {
  return x.size(dim);
}

template < typename S >
typename Tensor< S >::size_type Tensor< S >::length(const Tensor &x) {
  return x.length();
}

template < typename S >
typename Tensor< S >::stride_storage Tensor< S >::stride(const Tensor &x) {
  return x.stride();
}

template < typename S >
typename Tensor< S >::difference_type Tensor< S >::stride(
    const Tensor &x, dim_type dim) {
  return x.stride(dim);
}

template < typename S >
typename Tensor< S >::storage_pointer Tensor< S >::storage(const Tensor &x) {
  return x.storage();
}

template < typename S >
typename Tensor< S >::size_type Tensor< S >::offset(const Tensor &x) {
  return x.offset();
}

template < typename S >
typename Tensor< S >::pointer Tensor< S >::data(const Tensor &x) {
  return x.data();
}

template < typename S >
typename Tensor< S >::allocator_type Tensor< S >::allocator(const Tensor &x) {
  return x.allocator();
}

template < typename S >
typename Tensor< S >::reference Tensor< S >::get(const Tensor &x) {
  return x.get();
}

template < typename S >
typename Tensor< S >::reference Tensor< S >::get(
    const Tensor &x, size_type pos) {
  return x.get(pos);
}

template < typename S >
typename Tensor< S >::reference Tensor< S >::get(
    const Tensor &x, size_type pos0, size_type pos1) {
  return x.get(pos0, pos1);
}

template < typename S >
typename Tensor< S >::reference Tensor< S >::get(
    const Tensor &x, size_type pos0, size_type pos1, size_type pos2) {
  return x.get(pos0, pos1, pos2);
}

template < typename S >
typename Tensor< S >::reference Tensor< S >::get(
    const Tensor &x, size_type pos0, size_type pos1, size_type pos2,
    size_type pos3) {
  return x.get(pos0, pos1, pos2, pos3);
}

template < typename S >
typename Tensor< S >::reference Tensor< S >::get(
    const Tensor &x, const size_storage &pos) {
  return x.get(pos);
}

template < typename S >
bool Tensor< S >::isContiguous(const Tensor &x) {
  return x.isContiguous();
}

template < typename S >
bool Tensor< S >::partialContiguity(const Tensor &x, dim_type a, dim_type b) {
  return x.partialContiguity(a, b);
}

template < typename S >
bool Tensor< S >::isUnique(const Tensor &x) {
  return x.isUnique();
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_QUERY_HPP_
