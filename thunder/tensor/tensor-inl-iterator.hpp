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

#ifndef THUNDER_TENSOR_TENSOR_INL_ITERATOR_HPP_
#define THUNDER_TENSOR_TENSOR_INL_ITERATOR_HPP_

#include "thunder/tensor/tensor.hpp"
#include "thunder/tensor/tensor-inl.hpp"

#include <memory>
#include <utility>

#include "thunder/exception.hpp"

namespace thunder {
namespace tensor {

template< typename S >
Tensor< S >::iterator::iterator(const Tensor &x, size_type pos)
    : tensor_(&x), position_(pos) {}

template< typename S >
Tensor< S >::iterator::iterator(const iterator &it)
    : tensor_(it.tensor_), position_(it.position_) {}

template< typename S >
Tensor< S >::iterator::iterator(iterator &&it)
    : tensor_(::std::move(it.tensor_)), position_(::std::move(it.position_)) {}

template< typename S >
Tensor< S >::iterator::~iterator() {}

template< typename S >
typename Tensor< S >::iterator& Tensor< S >::iterator::operator=(iterator it) {
  ::std::swap(tensor_, it.tensor_);
  ::std::swap(position_, it.position_);
}

template< typename S >
bool Tensor< S >::iterator::operator==(const iterator& it) const {
  return tensor_ == it.tensor_ && position_ == it.position_;
}

template< typename S >
bool Tensor< S >::iterator::operator!=(const iterator& it) const {
  return tensor_ != it.tensor_ || position_ != it.position_;
}

template< typename S >
typename Tensor< S >::iterator& Tensor< S >::iterator::operator++() {
  ++position_;
  return *this;
}

template< typename S >
typename Tensor< S >::iterator Tensor< S >::iterator::operator++(int) {
  return iterator(*tensor_, position_++);
}

template< typename S >
Tensor< S >& Tensor< S >::iterator::operator*() const {
  current_ = (*tensor_)[position_];
  return current_;
}

template< typename S >
Tensor< S >* Tensor< S >::iterator::operator->() const {
  current_ = (*tensor_)[position_];
  return &current_;
}

template< typename S >
typename Tensor< S >::iterator Tensor< S >::begin() const {
  return iterator(*this);
}

template< typename S >
typename Tensor< S >::iterator Tensor< S >::end() const {
  return iterator(*this, size_[0]);
}

template< typename S >
typename Tensor< S >::iterator Tensor< S >::begin(
    const Tensor &x) {
  return x.begin();
}

template< typename S >
typename Tensor< S >::iterator Tensor< S >::end(
    const Tensor &x) {
  return x.end();
}

template< typename S >
Tensor< S >::reference_iterator::reference_iterator(const Tensor &x)
    : tensor_(&x), position_(x.size_.size(), 0) {}

template< typename S >
Tensor< S >::reference_iterator::reference_iterator(
const Tensor &x, size_storage pos) : tensor_(&x) {
  ::std::swap(position_, pos);
}

template< typename S >
Tensor< S >::reference_iterator::reference_iterator(
    const reference_iterator& it)
    : tensor_(it.tensor_), position_(it.position_) {}

template< typename S >
Tensor< S >::reference_iterator::reference_iterator(reference_iterator &&it)
    : tensor_(it.tensor_), position_(std::move(it.position_)) {}

template< typename S >
typename Tensor< S >::reference_iterator&
Tensor< S >::reference_iterator::operator=(reference_iterator it) {
  tensor_ = it.tensor_;
  position_ = it.position_;
  return *this;
}

template< typename S >
bool Tensor< S >::reference_iterator::operator==(
    const reference_iterator& it) const {
  if (tensor_ != it.tensor_) {
    return false;
  }
  for (dim_type i = 0; i < position_.size(); ++i) {
    if (position_[i] != it.position_[i]) {
      return false;
    }
  }
  return true;
}

template< typename S >
bool Tensor< S >::reference_iterator::operator!=(
    const reference_iterator& it) const {
  if (tensor_ != it.tensor_) {
    return true;
  }
  for (dim_type i = 0; i < position_.size(); ++i) {
    if (position_[i] != it.position_[i]) {
      return true;
    }
  }
  return false;
}

template< typename S >
typename Tensor< S >::reference_iterator&
Tensor< S >::reference_iterator::operator++() {
  ++position_[position_.size() - 1];
  dim_type i = position_.size();
  while (i > 0 && position_[i] >= tensor_->size(i)) {
    position_[i] = 0;
    ++position_[--i];
  }
  return *this;
}

template< typename S >
typename Tensor< S >::reference_iterator
Tensor< S >::reference_iterator::operator++(int) {
  reference_iterator it(*this);
  ++position_[position_.size() - 1];
  dim_type i = position_.size();
  while (i > 0 && position_[i] >= tensor_->size(i)) {
    position_[i] = 0;
    ++position_[--i];
  }
  return it;
}

template< typename S >
typename Tensor< S >::reference
Tensor< S >::reference_iterator::operator*() const {
  return (*tensor_)(position_);
}

template< typename S >
typename Tensor< S >::pointer
Tensor< S >::reference_iterator::operator->() const {
  return &(*tensor_)(position_);
}

template< typename S >
typename Tensor< S >::reference_iterator Tensor< S >::reference_begin() const {
  return reference_iterator(*this);
}

template< typename S >
typename Tensor< S >::reference_iterator Tensor< S >::reference_end() const {
  size_storage pos(size_.size());
  for (dim_type i = 0; i < size_.size() - 1; ++i) {
    pos[i] = size_[i] - 1;
  }
  pos[size_.size() - 1] = size_[size_.size() - 1];
  return reference_iterator(*this, pos);
}

template< typename S >
typename Tensor< S >::reference_iterator Tensor< S >::reference_begin(
    const Tensor &x) {
  return x.reference_begin();
}

template< typename S >
typename Tensor< S >::reference_iterator Tensor< S >::reference_end(
    const Tensor &x) {
  return x.reference_end();
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_ITERATOR_HPP_
