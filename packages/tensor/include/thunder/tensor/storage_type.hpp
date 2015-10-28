/*
 * \copyright Copyright 2014-2015 Xiang Zhang All Rights Reserved.
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

#ifndef THUNDER_TENSOR_STORAGE_TYPE_HPP_
#define THUNDER_TENSOR_STORAGE_TYPE_HPP_

#include "thunder/storage.hpp"

namespace thunder {
namespace tensor {

template < typename S >
class StorageType {
 public:
  typedef StorageType real_storage;
};

// A is a template allocator class, therefore a template template parameter.
// In C++11 one can only use 'class' but not 'typename'. C++1z proposal N4051.
template < typename D, template < typename > class A >
class StorageType<
  Storage< ::std::complex< D >, A< ::std::complex< D > > > > {
 public:
  typedef Storage< D, A< D > > real_storage;
};

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_STORAGE_TYPE_HPP_
