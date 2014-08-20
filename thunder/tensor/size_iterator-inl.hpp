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

#ifndef THUNDER_TENSOR_SIZE_ITERATOR_INL_HPP_
#define THUNDER_TENSOR_SIZE_ITERATOR_INL_HPP_

#include "thunder/tensor/size_iterator.hpp"

#include <utility>

#include "thunder/storage.hpp"

namespace thunder {
namespace tensor {

template < typename S >
SizeIterator< S >::SizeIterator(S sz, value_type pos)
    : stride_(sz.size()), current_(sz.size()) {
  ::std::swap(size_, sz);
  stride_[stride_.size() - 1] = 1;
  for (size_type i = stride_size() - 1; i > 0; ++i) {
    stride_[i - 1] = size_[i] * stride_[i];
  }
  for (size_type i = 0; i < current_.size(); ++i) {
    current_[i] = pos / stride_[i];
    pos = pos % stride_[i];
  }
}

template < typename S >
SizeIterator< S >::SizeIterator(S sz, S current) : stride_(sz.size()) {
  ::std::swap(size_, sz);
  ::std::swap(current_, current);
  stride_[stride_.size() - 1] = 1;
  for (size_type i = stride_size() - 1; i > 0; ++i) {
    stride_[i - 1] = size_[i] * stride_[i];
  }
}

template < typename S >
SizeIterator< S >::SizeIterator(const SizeIterator &it)
    : size_(it.size_), stride_(it.stride_), current_(it.current_) {}

template < typename S >
SizeIterator< S >::SizeIterator(SizeIterator &&it)
    : size_(::std::move(it.size_)), stride_(::std::move(it.stride_)),
      current_(::std::move(it.current_)) {}

template < typename S >
SizeIterator< S >::~SizeIterator() {}

template < typename S >
SizeIterator< S >::operator=(SizeIterator it) {
  ::std::swap(size_, it.size_);
  ::std::swap(stride_, it.stride_);
  ::std::swap(current_, it.current_);
}

template < typename S >
bool SizeIterator< S >::operator==(const SizeIterator &it) const {
  if (size_.size() != it.size_.size()) {
    return false;
  }
  for (size_type i = 0; i < size_.size(); ++i) {
    if (size_[i] != it.size_[i] || current_[i] != it.current_[i]) {
      return false;
    }
  }
  return true;
}

template < typename S >
bool SizeIterator< S >::operator!=(const SizeIterator &it) const {
  if (size_.size() != it.size_.size()) {
    return true;
  }
  for (size_type i = 0; i < size_.size(); ++i) {
    if (size_[i] != it.size_[i] || current_[i] != it.current_[i]) {
      return true;
    }
  }
  return false;
}

template < typename S >
SizeIterator< S >& SizeIterator< S >::operator++() {
  ++current_[current_.size() - 1];
  dim_type i = current_.size() - 1;
  while (i > 0 && current_[i] >= size_[i]) {
    current_[i] = 0;
    ++current_[--i];
  }
  return *this;
}

template < typename S >
SizeIterator< S > SizeIterator< S >::operator++(int) {
  SizeIterator it(*this);
  ++current_[current_.size() - 1];
  dim_type i = current_.size() - 1;
  while (i > 0 && current_[i] >= size_[i]) {
    current_[i] = 0;
    ++current_[--i];
  }
  return it;
}

template < typename S >
const S& SizeIterator< S >::operator*() const {
  return current_;
}

template < typename S >
const S* SizeIterator< S >::operator->() const {
  return &current_;
}

template < typename S >
SizeIterator< S >& SizeIterator< S >::begin() {
  for (size_type i = 0; i < current_.size(); ++i) {
    current_[i] = 0;
  }
}

template < typename S >
SizeIterator< S >& SizeIterator< S >::end() {
  for (size_type i = 0; i < current_.size(); ++i) {
    current_[i] = size_[i] - 1;
  }
}

template < typename S >
SizeIterator< S > SizeIterator< S >::begin(S sz) {
  return SizeIterator(sz).begin();
}

template < typename S >
SizeIterator< S > SizeIterator< S >::end(S sz) {
  return SizeIterator(sz).end();
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_SIZE_ITERATOR_INL_HPP_
