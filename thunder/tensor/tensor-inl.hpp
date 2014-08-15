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

#ifndef THUNDER_TENSOR_TENSOR_INL_HPP_
#define THUNDER_TENSOR_TENSOR_INL_HPP_

#include "thunder/tensor/tensor.hpp"

#include <memory>

#include "thunder/exception.hpp"

namespace thunder {
namespace tensor {

template < typename S >
Tensor< S >::Tensor()
    : storage_(new S(1)), offset_(0), size_(1,1), stride_(1,1){}

template < typename S >
Tensor< S >::Tensor(const storage_pointer &s, size_type os)
    : storage_(s), offset_(os), size_(1, storage_.size() - offset_ + 1),
      stride_(1,1) {
  if (storage_ == nullptr) {
    throw invalid_argument("Storage is nullptr.");
  }
  if (offset_ >= storage_.size()) {
    throw out_of_range("Offset exceeds storage.");
  }
}

template < typename S >
Tensor< S >::Tensor(const size_storage &sz)
    : size_(sz), stride_(sz.size()), offset_(0) {
  if (size_.size() == 0) {
    throw invalid_argument("Size is empty.");
  }
  size_type storage_size = 1;
  for (const size_type &size_x : size_) {
    if (size_x == 0) {
      throw invalid_argument("Size evaluates to zero.");
    }
    storage_size *= size_x;
  }
  storage_ = ::std::make_shared< S >(storage_size);
  stride_[stride_.size() - 1] = 1;
  for (dim_type i = stride_.size() - 1; i > 0; --i) {
    stride_[i - 1] = size_[i] * stride_[i];
  }
}

template < typename S >
Tensor< S >::Tensor(size_type sz0) : Tensor({sz0}) {}

template < typename S >
Tensor< S >::Tensor(size_type sz0, size_type sz1) : Tensor({sz0, sz1}) {}

template < typename S >
Tensor< S >::Tensor(size_type sz0, size_type sz1, size_type sz2)
    : Tensor({sz0, sz1, sz2}) {}

template < typename S >
Tensor< S >::Tensor(size_type sz0, size_type sz1, size_type sz2, size_type sz3)
    : Tensor({sz0, sz1, sz2, sz3}) {}

template < typename S >
Tensor< S >::Tensor(const size_storage &sz, const storage_pointer &s,
                    size_type os)
    : size_(sz), stride_(sz.size()), storage_(s), offset_(os) {
  if (storage_ == nullptr) {
    throw invalid_argument("Storage is nullptr.");
  }
  if (size_.size() == 0) {
    throw invalid_argument("Size is empty.");
  }
  size_type storage_size = 1;
  for (const size_type &size_x : size_) {
  if (size_x == 0) {
    throw invalid_argument("Size evaluates to zero.");
  }
    storage_size *= size_x;
  }
  if (storage_size + offset > storage_.size()) {
    throw out_of_range("Offset and size exceed storage size.");
  }
  stride_[stride_.size() - 1] = 1;
  for (dim_type i = stride_.size() - 1; i > 0; --i) {
    stride_[i - 1] = size_[i] * stride_[i];
  }
}

template< typename S >
Tensor< S >::Tensor(const size_storage &sz, const stride_storage &st,
                    const storage_pointer &s, size_type os)
    :size_(sz), stride_(st), storage_(s), offset_(os){
  if (storage_ == nullptr) {
    throw invalid_argument("Storage is nullptr.");
  }
  if (size_.size() == 0) {
    throw invalid_argument("Size is empty.");
  }
  if (stride_.size() == 0) {
    throw invalid_argument("Stride is empty.");
  }
  if (size_.size() != stride_.size()) {
    throw invalid_argument("Size and stride have different length.");
  }
  for (const size_type &size_x : size_) {
    if (size_x == 0) {
      throw invalid_argument("Size evaluates to zero.");
    }
  }
  difference_type max_offset = offset_, min_offset = offset_;
  for (dim_type i = 0; i < size_.size(); ++i) {
    if (stride_[i] < 0) {
      min_offset = min_offset + (size_[i]-1)*stride_[i];
    } else if (stride_[i] > 0) {
      max_offset = max_offset + (size_[i]-1)*stride_[i];
    }
  }
  if (min_offset < 0 || max_offset >= storage_.size()) {
    throw length_error("Offset, size and stride exceed storage size.");
  }
}

template < typename S >
Tensor< S >::Tensor(const Tensor &y)
    : size_(y.size_), stride_(y.stride_), storage_(y.storage_),
      offset_(y.offset_) {}

template < typename S >
Tensor< S >::Tensor(Tensor &&y)
    : size_(std::move(y.size_)), stride_(std::move(y.stride_)),
      storage_(std::move(y.storage_)), offset_(std::move(y.offset_)) {}

template < typename S >
Tensor< S >::~Tensor() {}

template < typename S >
template < typename T >
bool Tensor< S >::isSameSizeAs(const T &y) const{
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
bool Tensor< S >::isSameSizeAs(const Tensor &y) const{
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

#endif  // THUNDER_TENSOR_TENSOR_INL_HPP
