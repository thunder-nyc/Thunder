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

#ifndef THUNDER_TENSOR_SIZE_ITERATOR_HPP_
#define THUNDER_TENSOR_SIZE_ITERATOR_HPP_

#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "thunder/storage.hpp"

namespace thunder {
namespace tensor {

template < typename S >
class SizeIterator {
 public:
  SizeIterator(S sz);
  SizeIterator(const SizeIterator &it);
  SizeIterator(SizeIterator &&it);
  ~SizeIterator();

  SizeIterator& operator=(SizeIterator it);

  bool operator==(const SizeIterator &it) const;
  bool operator!=(const SizeIterator &it) const;

  SizeIterator& operator++();
  SizeIterator operator++(int);

  const S& operator*() const;
  const S* operator->() const;

  SizeIterator& begin();
  SizeIterator& end();

  static SizeIterator begin(S sz);
  static SizeIterator end(S sz);

  
 private:
  S size_;
  S current_;
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_SIZE_ITERATOR_HPP_
