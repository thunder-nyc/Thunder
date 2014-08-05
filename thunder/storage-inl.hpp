/*
 * \copyright Copyright 2014 Xiang Zhang. All Rights Reserved.
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

#ifndef THUNDER_STORAGE_INL_HPP
#define THUNDER_STORAGE_INL_HPP

#include "thunder/storage.hpp"

#include <initializer_list>
#include <memory>
#include <stdexcept>

namespace thunder {

template <typename T, typename A >
Storage<T, A>::Storage(const A &alloc = A())
    : alloc_(alloc), size_(0), data_(nullptr) {}

template <typename T, typename A >
Storage<T, A>::Storage(size_type count, const A &alloc = A())
    : alloc_(alloc), size_(count),
      data_(size_ == 0 ? nullptr : alloc_.allocate(size_)){}

template <typename T, typename A >
Storage<T, A>::Storage(size_type count, const T& value, const A &alloc = A())
    : alloc_(alloc), size_(count),
      data_(size_ == 0 ? nullptr : alloc_.allocate(size_)){
  for(size_type i = 0; i < size_; ++i) {
    data_[i] = value;
  }
}

template <typename T, typename A >
Storage<T, A>::Storage(const Storage &other)
  : alloc_(other.alloc_), size_(other.size_),
    data_(size_ == 0 ? nullptr : alloc_.allocate(size_)) {
  for(size_type i = 0; i < size_; ++i) {
    data_[i] = other.data_[i];
  }
}

template <typename T, typename A >
Storage<T, A>::Storage(Storage &&other)
    : alloc_(::std::move(other.alloc_)), size_(::std::move(other.size_)),
      data_(other.data_) {
  other.data_ = nullptr;
}

template <typename T, typename A >
Storage<T, A>::~Storage() {
  if (data_ != nullptr) {
    alloc_.deallocate(data_, size_);
  }
}

template <typename T, typename A >
Storage<T, A>& Storage<T,A>::operator=(const Storage &other) {
  Resize(other.size_);
  for(size_type i = 0; i < size_; ++i) {
    data_[i] = other.data_[i];
  }
  return *this;
}

template <typename T, typename A >
Storage<T, A>& Storage<T,A>::operator=(Storage && other) {
  if (data_ != nullptr) {
    alloc_.deallocate(data_, size_);
  }
  size_ = ::std::move(other.size_);
  data_ = other.data_;
  other.data_ = nullptr;
  return *this;
}

template <typename T, typename A >
Storage<T,A>::reference Storage<T,A>::operator[](size_type pos) {
  return data_[pos];
}

template <typename T, typename A >
Storage<T,A>::const_reference Storage<T,A>::operator[](size_type pos) const {
  return data_[pos];
}

template <typename T, typename A >
Storage<T,A>::pointer Storage<T,A>::Data() {
  return data_;
}

template <typename T, typename A >
Storage<T,A>::const_pointer Storage<T,A>::Data() const {
  return data_;
}

template <typename T, typename A >
Storage<T,A>::iterator Storage<T,A>::begin() {
  return data_;
}

template <typename T, typename A >
Storage<T,A>::const_iterator Storage<T,A>::begin() const {
  return data_;
}

template <typename T, typename A >
Storage<T,A>::iterator Storage<T,A>::end() {
  return data_ + size_;
}

template <typename T, typename A >
Storage<T,A>::const_iterator Storage<T,A>::end() const {
  return data_ + size_;
}

template<typename T, typename A >
template<typename Other_T, typename Other_A>
void Storage<T,A>::Copy(const Storage<Other_T, Other_A> &other) {
  Resize(static_cast<size_type>(other.Size()));
  for(size_type i = 0; i < size_; ++i) {
    data_[i] = static_cast<T> (other[static_cast<other::size_type>(i)]);
  }
}

template <typename T, typename A >
void Storage<T,A>::Resize(size_type count) {
  pointer data = nullptr;
  if (count > 0) {
    data = alloc_.allocate(count, data_);
  }
  if (data_ != nullptr) {
    alloc_.deallocate(data_, size_);
  }
  size_ = count;
  data_ = data;
}

template <typename T, typename A >
void Storage<T,A>::Resize(size_type count, const T &value) {
  Resize(count);
  for(size_type i = 0; i < size_; ++i) {
    data_[i] = value;
  }
}

template <typename T, typename A >
Storage<T,A>::size_type Storage<T,A>::Size() const {
  return size_;
}

}  // namespace thunder

#endif  // THUNDER_STORAGE_INL_HPP
