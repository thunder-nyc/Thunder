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

#ifndef THUNDER_STORAGE_HPP_
#define THUNDER_STORAGE_HPP_

#include <initializer_list>
#include <memory>

namespace thunder {

template <typename T, typename Allocator = std::allocator<T> >
class Storage {
 public:
  // Typedefs from allocator
  typedef T value_type;
  typedef Allocator allocator_type;
  typedef typename Allocator::reference reference;
  typedef typename Allocator::const_reference const_reference;
  typedef typename Allocator::difference_type difference_type;
  typedef typename Allocator::size_type size_type;
  typedef std::allocator_traits<Allocator>::pointer pointer;
  typedef std::allocator_traits<Allocator>::const_pointer const_pointer;

  // Iterator definitions
  typedef pointer iterator;
  typedef const_pointer const_iterator;
  typedef std::reverse_iterator<iterator> reverse_iterator;
  typedef std::reverse_iterator<const_iterator> const_reverse_iterator;

  // Default Constructor
  explicit Storage(const Allocator &alloc = Allocator());
  // Constructor with given size
  explicit Storage(size_type count, const Allocator &alloc = Allocator());
  // Constructor with given size and a default value
  explicit Storage(size_type count, const T &value,
                   const Allocator &alloc = Allocator());
  // Copy constructor
  Storage(const Storage &other);
  // Move constructor
  Storage(Storage &&other);

  // Destructor
  virtual ~Storage();

  // Assignment operator
  Storage &operator=(const Storage &other);
  // Move assignment operator
  Storage &operator=(Storage && other);

  // Get reference at pos with bound checking.
  reference At(size_type pos);
  // Get const reference at pos with bound checking.
  const_reference At(size_type pos) const;
  // Get reference at pos without bound checking
  reference operator[](size_type pos);
  // Get const reference at pos without bound checking
  const_reference operator[](size_type pos);

  // Get raw pointer to data
  pointer Data();
  // Get const raw pointer to data
  const_pointer Data() const;

  // begin() and end() functions does not follow style guide for a reason: they
  // enables ranged for iterators like this:
  // for(auto &val : storage) {/* Do something here with val */}
  // Get iterator to data
  iterator begin();
  // Get const iterator to data
  const_iterator begin() const;
  // Get iterater pass the last element
  iterator end();
  // Get const iterator passing the last element
  const_iterator end() const;

  // Copy from a different storage. Precision may be lost
  template<typename Other_T, Other_Allocator>
  void Copy(const Storage<Other_T, Other_Allocator> &other);
  // Resize. Data content will be lost.
  void Resize(std::size_t count);
  // Resize with all elements using target value
  void Resize(std::size_t count, const T &value);
  // Check the size of the storage
  size_type Size() const;

 private:
  Allocator alloc_;
  pointer data_;
  size_type size_;
};

}  // namespace thunder

#endif  // THUNDER_STORAGE_HPP_
