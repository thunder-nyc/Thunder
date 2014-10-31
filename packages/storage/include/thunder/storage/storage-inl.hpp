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

#ifndef THUNDER_STORAGE_STORAGE_INL_HPP
#define THUNDER_STORAGE_STORAGE_INL_HPP

#include "thunder/serializer.hpp"
#include "thunder/storage/storage.hpp"

#include <memory>
#include <utility>

namespace thunder {
namespace storage {

template < typename D, typename A >
Storage< D, A >::Storage(const A &alloc)
    : alloc_(alloc), size_(0), data_(nullptr) {}

template < typename D, typename A >
Storage< D, A >::Storage(size_type count, const A &alloc)
    : alloc_(alloc), size_(count),
      data_(size_ == 0 ? nullptr : alloc_.allocate(size_)) {}

template < typename D, typename A >
Storage< D, A >::Storage(size_type count, const_reference value, const A &alloc)
    : alloc_(alloc), size_(count),
      data_(size_ == 0 ? nullptr : alloc_.allocate(size_)) {
  for (size_type i = 0; i < size_; ++i) {
    data_[i] = value;
  }
}

template < typename D, typename A >
Storage< D, A >::Storage(const Storage &other)
    : alloc_(other.alloc_), size_(other.size_),
      data_(size_ == 0 ? nullptr : alloc_.allocate(size_)) {
  for (size_type i = 0; i < size_; ++i) {
    data_[i] = other.data_[i];
  }
}

template < typename D, typename A >
Storage< D, A >::Storage(Storage &&other)
    : alloc_(::std::move(other.alloc_)), size_(::std::move(other.size_)),
      data_(other.data_) {
  other.data_ = nullptr;
}

template < typename D, typename A >
Storage< D, A >::Storage(::std::initializer_list< D > init, const A& alloc)
    :alloc_(alloc), size_(init.size()),
     data_(size_ == 0 ? nullptr : alloc_.allocate(size_)) {
  size_t i = 0;
  for (const D& value : init) {
    data_[i++] = value;
  }
}

template < typename D, typename A >
Storage< D, A >::~Storage() {
  if (data_ != nullptr) {
    alloc_.deallocate(data_, size_);
  }
}

template < typename D, typename A >
Storage< D, A > &Storage< D, A >::operator=(Storage< D, A > other) {
  std::swap(size_, other.size_);
  std::swap(data_, other.data_);
  std::swap(alloc_, other.alloc_);
  return *this;
}

template < typename D, typename A >
typename Storage< D, A >::reference Storage< D, A >::operator[](size_type pos) {
  return data_[pos];
}

template < typename D, typename A >
typename Storage< D, A >::const_reference Storage< D, A >::operator[](
    size_type pos) const {
  return data_[pos];
}

template < typename D, typename A >
typename Storage< D, A >::pointer Storage< D, A >::data() {
  return data_;
}

template < typename D, typename A >
typename Storage< D, A >::const_pointer Storage< D, A >::data() const {
  return data_;
}

template < typename D, typename A >
typename Storage< D, A >::iterator Storage< D, A >::begin() {
  return data_;
}

template < typename D, typename A >
typename Storage< D, A >::const_iterator Storage< D, A >::begin() const {
  return data_;
}

template < typename D, typename A >
typename Storage< D, A >::iterator Storage< D, A >::end() {
  return data_ + size_;
}

template < typename D, typename A >
typename Storage< D, A >::const_iterator Storage< D, A >::end() const {
  return data_ + size_;
}

template < typename D, typename A >
template < typename S >
void Storage< D, A >::copy(const S &other) {
  if (this != reinterpret_cast<const Storage*> (&other)) {
    resize(static_cast<size_type>(other.size()));
    for (size_type i = 0; i < size_; ++i) {
      data_[i] = static_cast< D > (
          other[static_cast< typename S::size_type >(i)]);
    }
  }
}

template < typename D, typename A >
void Storage< D, A >::resize(size_type count) {
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

template < typename D, typename A >
void Storage< D, A >::resize(size_type count, const_reference value) {
  resize(count);
  for (size_type i = 0; i < size_; ++i) {
    data_[i] = value;
  }
}

template < typename D, typename A >
typename Storage< D, A >::size_type Storage< D, A >::size() const {
  return size_;
}

}  // namespace storage
}  // namespace thunder

namespace thunder {
namespace serializer {

template < typename S, typename D, typename A >
void save(S *s, const storage::Storage< D, A > &t) {
  typedef storage::Storage< D, A > T;

  // Save size of storage
  typename T::size_type size = t.size();
  s->save(size);

  // Save data of storage
  typename T::const_pointer data = t.data();
  for (typename T::size_type i = 0; i < size; ++i) {
    s->save(data[i]);
  }
}

template < typename S, typename D, typename A >
void load(S *s, storage::Storage< D, A > *t) {
  typedef storage::Storage< D, A > T;

  // Load size of storage
  typename T::size_type size;
  s->load(&size);
  t->resize(size);

  // Restore data of storage
  typename T::pointer data = t->data();
  for (typename T::size_type i = 0; i < size; ++i) {
    s->load(&data[i]);
  }
}

}  // namespace serializer
}  // namespace thunder

#endif  // THUNDER_STORAGE_STORAGE_INL_HPP
