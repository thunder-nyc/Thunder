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

// Default constructor
template <typename T, typename Allocator = std::allocator<T> >
Storage <T, Allocator>::Storage(const Allocator &alloc = Allocator())
    : alloc_(alloc), size_(0), data_(nullptr) {}

// Constructor to default size
template <typename T, typename Allocator = std::allocator<T> >
Storage<T, Allocator>::Storage(size_type count,
                               const Allocator &alloc = Allocator())
    : alloc_(alloc), size_(count), data_(alloc_.allocate(size_)){}

// Constructor with default value
template <typename T, typename Allocator = std::allocator<T> >
Storage<T, Allocator>::Storage(size_type count,
                               const T& value,
                               const Allocator &alloc = Allocator())
    : alloc_(alloc), size_(count), data_(alloc_.allocate(size_)){
  for(size_type i = 0; i < size_; ++i) {
    data_[i] = value;
  }
}

// Copy constructor
template <typename T, typename Allocator = std::allocator<T> >
Storage<T, Allocator>::Storage(const Storage &other)
    : alloc_(other.alloc_), size_(other.size_), data_(alloc_.allocate(size_)) {
  for(size_type i = 0; i < size_; ++i) {
    data_[i] = other.data_[i];
  }
}

// Move constructor
template <typename T, typename Allocator = std::allocator<T> >
Storage<T, Allocator>::Storage(Storage &&other)
    : alloc_(std::move(other.alloc_)), size_(std::move(other.size_)),
      data_(other.data_) {
  other.data_ = nullptr;
}

// Destructor
template <typename T, typename Allocator = std::allocator<T> >
Storage<T, Allocator>::~Storage() {
  if (data_ != nullptr) {
    alloc_.deallocate(data_, size_);
  }
}

// Assignment operator
Storage &operator=(const Storage &other) {
  Resize(other.size_);
  for(size_type i = 0; i < size_; ++i) {
    data_[i] = other.data_[i];
  }
}

// Move assignment operator
Storage &operator=(Storage && other) {
  if (data_ != nullptr) {
    alloc_.deallocate(data_, size_);
  }
  size_ = std::move(other.size_);
  data_ = other.data_;
  other.data_ = nullptr;
}

// Get reference at pos without bound checking
reference operator[](size_type pos) {
  return data_[pos];
}

// Get const reference at pos without bound checking
const_reference operator[](size_type pos) {
  return data_[pos];
}

// Get raw pointer to data
pointer Data() {
  return data_;
}

// Get const raw pointer to data
const_pointer Data() const {
  return data_;
}

// Get iterator to data
iterator begin() {
  return data_;
}
// Get const iterator to data
const_iterator begin() const {
  return data_;
}

// Get iterater pass the last element
iterator end() {
  return data_ + size_;
}

// Get const iterator passing the last element
const_iterator end() const {
  return data_ + size_;
}

// Copy from a different storage. Precision may be lost.
template<typename Other_T, Other_Allocator>
void Copy(const Storage<Other_T, Other_Allocator> &other) {
  Resize(static_cast<size_type>(other.size_));
  for(size_type i = 0; i < size_; ++i) {
    data_[i] = static_cast<T> other[static_cast<other::size_type>(i)];
  }
}

// Resize. Data content will be lost.
void Resize(size_type count) {
  pointer data = alloc_.allocate(count, data_);
  if (data_ != nullptr) {
    alloc_.deallocate(data_, size_);
  }
  size_ = count;
  data_ = data;
}

// Resize with all elements using target value
void Resize(size_type count, const T &value) {
  Resize(count);
  for(size_type i = 0; i < size_; ++i) {
    data_[i] = value;
  }
}

// Check the size of the storage
size_type Size() const {
  return size_;
}


}  // namespace thunder

#endif  // THUNDER_STORAGE_INL_HPP
