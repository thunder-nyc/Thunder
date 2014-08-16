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

#ifndef THUNDER_TENSOR_TENSOR_INL_QUERIES_HPP_
#define THUNDER_TENSOR_TENSOR_INL_QUERIES_HPP_

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
bool Tensor< S >::isSameSizeAs(const Tensor &y) const {
  if (size_.size() != y.size_.size()) {
    return false;
  }
  for (dim_type i = 0; i < size_.size(); ++i) {
    if (size_[i] != y.size_[i]) {
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
bool Tensor< S >::isSameSizeAs(const Tensor &x, const Tensor &y) {
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
typename Tensor< S >::size_type Tensor< S >::count() const {
  size_type count = 1;
  for (dim_type i = 0; i < size_.size(); ++i) {
    count *= size_[i];
  }
  return count;
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
bool Tensor< S >::isContiguous() const {
  if (stride_[stride_.size() - 1] != 1) {
    return false;
  }
  for (dim_type i = stride_.size() - 1; i > 0; --i) {
    if (stride_[i - 1] != stride_[i]*size_[i]) {
      return false;
    }
  }
  return true;
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
typename Tensor< S >::size_type Tensor< S >::count(const Tensor &x) {
  return x.count();
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
bool Tensor< S >::isContiguous(const Tensor &x) {
  return x.isContiguous();
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_QUERIES_HPP_
