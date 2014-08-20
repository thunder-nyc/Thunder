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

#ifndef THUNDER_TENSOR_INDEX_ITERATOR_HPP_
#define THUNDER_TENSOR_INDEX_ITERATOR_HPP_

#include "thunder/storage.hpp"

namespace thunder {
namespace tensor {

template < typename S >
class IndexIterator {
 public:
  // Iterator tag
  typedef std::input_iterator_tag iterator_category;

  // Typedefs from S
  typedef S storage_type;
  typedef typename S::allocator_type allocator_type;
  typedef typename S::value_type value_type;
  typedef typename S::reference reference;
  typedef typename S::const_reference const_reference;
  typedef typename S::difference_type difference_type;
  typedef typename S::size_type size_type;
  typedef typename S::pointer pointer;
  typedef typename S::const_pointer const_pointer;

  explicit IndexIterator(S sz, value_type pos = 0);
  IndexIterator(S sz, S current);
  IndexIterator(const IndexIterator &it);
  IndexIterator(IndexIterator &&it);
  ~IndexIterator();

  IndexIterator& operator=(IndexIterator it);

  bool operator==(const IndexIterator &it) const;
  bool operator!=(const IndexIterator &it) const;

  IndexIterator& operator++();
  IndexIterator operator++(int);

  const S& operator*() const;
  const S* operator->() const;

  IndexIterator& begin();
  IndexIterator& end();

  static IndexIterator begin(S sz);
  static IndexIterator end(S sz);

 private:
  S size_;
  S stride_;
  S current_;
};

}  // namespace tensor
}  // namespace thunder

#include "thunder/tensor/index_iterator-inl.hpp"

#endif  // THUNDER_TENSOR_INDEX_ITERATOR_HPP_
