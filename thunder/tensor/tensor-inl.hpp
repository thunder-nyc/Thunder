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

template < typename S >
Tensor< S >::Tensor(const storage_pointer &s, size_type os)
    : storage_(s), offset_(os) {
  if (storage_ == nullptr) {
    return;
  }
  if (offset_ >= storage_.size()) {
    throw exception::out_of_range("Offset exceeds storage.");
  }
  size_.resize(1, storage_.size() - offset_ + 1);
  stride_.resize(1, 1);
}

template < typename S >
Tensor< S >::Tensor(const size_storage &sz)
    : size_(sz), stride_(sz.size(), 1), storage_(nullptr), offset_(0) {
  size_type storage_size = 1;
  if (size_.size() == 0) {
    return;
  }
  for (const size_type &size_x : size_) {
    storage_size *= size_x;
  }
  if (storage_size == 0) {
    size_.resize(0);
    stride_.resize(0);
    return;
  }
  storage_ = ::std::make_shared< S >(storage_size);
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
    : size_(sz), stride_(sz.size(),1), storage_(s), offset_(os) {
  size_type storage_size = 1;
  for (const size_type &size_x : size_) {
    storage_size *= size_x;
  }
  if (storage_ == nullptr || size_.size() == 0 || storage_size == 0) {
    storage_ = nullptr;
    size_.resize(0);
    stride_.resize(0);
    offset_ = 0;
    return;
  }
  if (storage_size + offset > storage_.size()) {
    throw exception::out_of_range("Offset and size exceed storage.");
  }
}

#endif  // THUNDER_TENSOR_TENSOR_INL_HPP
