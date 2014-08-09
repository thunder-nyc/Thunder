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

template < typename S, typename I >
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

  // Declaration of iterators
  class iterator;
  class const_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  // Constructors

 private:
  thunder::Storage< size_type, ::std::allocator<T> > size_;
  thunder::Storage< size_type, ::std::allocator<T> > stride_;
  ::std::shared_ptr< S > storage_;
  size_type offset_;
}

}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_HPP
