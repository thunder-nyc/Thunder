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

#ifndef THUNDER_STORAGE_STORAGE_HPP_
#define THUNDER_STORAGE_STORAGE_HPP_

#include <memory>

namespace thunder {

template < typename T, typename A = ::std::allocator<T>, int S = 0 >
class Storage {
 public:
  // Typedefs from allocator
  typedef A allocator_type;
  typedef typename A::value_type value_type;
  typedef typename A::reference reference;
  typedef typename A::const_reference const_reference;
  typedef typename A::difference_type difference_type;
  typedef typename A::size_type size_type;
  typedef typename ::std::allocator_traits<A>::pointer pointer;
  typedef typename ::std::allocator_traits<A>::const_pointer const_pointer;

  // Iterator definitions
  typedef pointer iterator;
  typedef const_pointer const_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  // Default Constructor
  explicit Storage(const A &alloc = A());
  // Constructor with given size
  explicit Storage(size_type count, const A &alloc = A());
  // Constructor with given size and a default value
  explicit Storage(size_type count, const_reference value,
                   const A &alloc = A());
  // Copy constructor
  Storage(const Storage &other);
  // Move constructor
  Storage(Storage &&other);

  // Destructor
  ~Storage();

  // Assignment operator (using copy and swap idiom)
  Storage &operator=(Storage other);

  // Get reference at pos without bound checking
  reference operator[](size_type pos);
  // Get const reference at pos without bound checking
  const_reference operator[](size_type pos) const;

  // Get raw pointer to data
  pointer Data();
  // Get const raw pointer to data
  const_pointer Data() const;

  // Get iterator to data
  iterator begin();
  // Get const iterator to data
  const_iterator begin() const;
  // Get iterater pass the last element
  iterator end();
  // Get const iterator passing the last element
  const_iterator end() const;

  // Copy from a different storage using static casts
  template< typename Other_T, typename Other_A, int Other_S >
  void Copy(const Storage< Other_T, Other_A, Other_S > &other);

  // Resize. Data content will be lost.
  void Resize(size_type count);
  // Resize with all elements using target value
  void Resize(size_type count, const_reference value);

  // Check the size of the storage
  size_type Size() const;

 private:
  A alloc_;
  size_type size_;
  pointer data_;
};

}  // namespace thunder

#endif  // THUNDER_STORAGE_STORAGE_HPP_
