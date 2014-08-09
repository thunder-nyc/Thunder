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

#ifndef THUNDER_TENSOR_TENSOR_HPP
#define THUNDER_TENSOR_TENSOR_HPP

#include <initializer_list>
#include <memory>

#include "thunder/storage.hpp"

namespace thunder{

template < typename S >
class Tensor {
 public:
  // Typedefs from storage
  typedef typename S::allocator_type allocator_type;
  typedef typename S::value_type value_type;
  typedef typename S::reference reference;
  typedef typename S::const_reference const_reference;
  typedef typename S::difference_type difference_type;
  typedef typename S::size_type size_type;
  typedef typename S::pointer pointer;
  typedef typename S::const_pointer const_pointer;

  // Typedefs for tensor
  typedef Storage< size_type > size_storage;
  typedef Storage< difference_type > stride_storage;
  typedef ::std::shared_ptr< S > storage_pointer;

  // Declaration of iterator class
  class iterator;
  // Declaration of const_iterator class
  class const_iterator;

  // Default constructor
  explicit Tensor(const storage_pointer &storage = storage_pointer(new S()),
                  const size_type offset = 0);
  // Explicit size constructor support up to 8 dimensions
  explicit Tensor(const size_type &size0, const size_type &size1 = 0,
                  const size_type &size2 = 0, const size_type &size3 = 0,
                  const size_type &size4 = 0, const size_type &size5 = 0,
                  const size_type &size6 = 0, const size_type &size7 = 0,
                  const storage_pointer &storage = storage_pointer(new S()),
                  const size_type offset = 0);
  // Explicit size constructor using storage class
  explicit Tensor(const size_storage &size,
                  const storage_pointer &storage = storage_pointer(new S()),
                  const size_type offset = 0);
  // Explicit size and stride constructor using storage class
  explicit Tensor(const size_storage &size, const stride_storage &stride,
                  const storage_pointer &storage = storage_pointer(new S()),
                  const size_type offset = 0);

  // Copy constructor
  Tensor(const Tensor &other);
  // Move constructor
  Tensor(Tensor &&other);

  // Make destructor virtual for supporting polymorphism.
  virtual ~Tensor();

  // Assignment operator using copy and swap idiom
  virtual Tensor &operator=(Tensor other);

  // Index operators
  virtual Tensor operator[](const size_type &index);
  virtual Tensor operator[](const ::std::initializer_list< size_type > &index);

  // Apply a lambda. lambda must return value_type or references to value_type.
  template < typename L >
  virtual Tensor& Apply(L &lambda);

 private:
  size_storage size_;
  stride_storage stride_;
  storage_pointer storage_;
  size_type offset_;
};

}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_HPP
