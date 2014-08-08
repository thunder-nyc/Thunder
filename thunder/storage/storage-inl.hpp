/*
 * Copyright 2014 Xiang Zhang. All Rights Reserved.
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
 */

#ifndef THUNDER_STORAGE_STORAGE_INL_HPP
#define THUNDER_STORAGE_STORAGE_INL_HPP

#include "thunder/storage/storage.hpp"

#include <memory>
#include <utility>

namespace thunder {

template < typename T, typename A, int S >
Storage< T, A, S >::Storage(const A &alloc)
    : alloc_(alloc), size_(0), data_(nullptr) {}

template < typename T, typename A, int S >
Storage< T, A, S >::Storage(Storage< T, A, S >::size_type count, const A &alloc)
    : alloc_(alloc), size_(count),
      data_(size_ == 0 ? nullptr : alloc_.allocate(size_)) {}

template < typename T, typename A, int S >
Storage< T, A, S >::Storage(Storage< T, A, S >::size_type count,
                       Storage< T, A, S >::const_reference value,
                       const A &alloc)
    : alloc_(alloc), size_(count),
      data_(size_ == 0 ? nullptr : alloc_.allocate(size_)) {
  for (size_type i = 0; i < size_; ++i) {
    data_[i] = value;
  }
}

template < typename T, typename A, int S >
Storage< T, A, S >::Storage(const Storage &other)
    : alloc_(other.alloc_), size_(other.size_),
      data_(size_ == 0 ? nullptr : alloc_.allocate(size_)) {
  for (size_type i = 0; i < size_; ++i) {
    data_[i] = other.data_[i];
  }
}

template < typename T, typename A, int S >
Storage< T, A, S >::Storage(Storage &&other)
    : alloc_(::std::move(other.alloc_)), size_(::std::move(other.size_)),
      data_(other.data_) {
  other.data_ = nullptr;
}

template <typename T, typename A, int S >
Storage< T, A, S >::~Storage() {
  if (data_ != nullptr) {
    alloc_.deallocate(data_, size_);
  }
}

template < typename T, typename A, int S >
Storage< T, A, S > &Storage< T, A, S >::operator=(Storage< T, A, S > other) {
  std::swap(size_, other.size_);
  std::swap(data_, other.data_);
  std::swap(alloc_, other.alloc_);
  return *this;
}

template < typename T, typename A, int S >
typename Storage< T, A, S >::reference Storage< T, A, S >::operator[](
    Storage< T, A, S >::size_type pos) {
  return data_[pos];
}

template <typename T, typename A, int S >
typename Storage< T, A, S >::const_reference Storage< T, A, S >::operator[](
    Storage< T, A, S >::size_type pos) const {
  return data_[pos];
}

template < typename T, typename A, int S >
typename Storage< T, A, S >::pointer Storage< T, A, S >::Data() {
  return data_;
}

template < typename T, typename A, int S >
typename Storage< T, A, S >::const_pointer Storage< T, A, S >::Data() const {
  return data_;
}

template < typename T, typename A, int S >
typename Storage< T, A, S >::iterator Storage< T, A, S >::begin() {
  return data_;
}

template < typename T, typename A, int S >
typename Storage< T, A, S >::const_iterator Storage< T, A, S >::begin() const {
  return data_;
}

template < typename T, typename A, int S >
typename Storage< T, A, S >::iterator Storage< T, A, S >::end() {
  return data_ + size_;
}

template < typename T, typename A, int S >
typename Storage< T, A, S >::const_iterator Storage< T, A, S >::end() const {
  return data_ + size_;
}

template< typename T, typename A, int S >
template< typename Other_T, typename Other_A, int Other_S >
void Storage< T, A, S >::Copy(
    const Storage< Other_T, Other_A, Other_S > &other) {
  if (this != reinterpret_cast<const Storage*> (&other)) {
    Resize(static_cast<size_type>(other.Size()));
    for (size_type i = 0; i < size_; ++i) {
      data_[i] = static_cast<T> (
          other[static_cast<
                typename Storage< Other_T, Other_A, Other_S >::size_type >(i)]);
    }
  }
}

template < typename T, typename A, int S >
void Storage< T, A, S >::Resize(Storage< T, A, S >::size_type count) {
  if (size_ != count) {
    if (data_ != nullptr) {
      alloc_.deallocate(data_, size_);
    }
    pointer data = nullptr;
    if (count > 0) {
      data = alloc_.allocate(count);
    }
    size_ = count;
    data_ = data;
  }
}

template < typename T, typename A, int S >
void Storage< T, A, S >::Resize(Storage< T, A, S >::size_type count,
                           Storage< T, A, S >::const_reference value) {
  Resize(count);
  for (size_type i = 0; i < size_; ++i) {
    data_[i] = value;
  }
}

template <typename T, typename A, int S >
typename Storage< T, A, S >::size_type Storage< T, A, S >::Size() const {
  return size_;
}

}  // namespace thunder

#endif  // THUNDER_STORAGE_STORAGE_INL_HPP
