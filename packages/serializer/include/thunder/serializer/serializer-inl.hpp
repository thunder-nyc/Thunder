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

#ifndef THUNDER_SERIALIZER_SERIALIZER_INL_HPP_
#define THUNDER_SERIALIZER_SERIALIZER_INL_HPP_

#include "thunder/serializer/serializer.hpp"

#include <memory>
#include <sstream>

#include "thunder/serializer/binary_serializer.hpp"
#include "thunder/serializer/text_serializer.hpp"

namespace thunder {
namespace serial {

template < typename M >
Serializer< M >::Serializer(stream_pointer m) : stream_(m) {}

template < typename M >
Serializer< M >::~Serializer() {}

template < typename M >
typename Serializer< M >::stream_pointer stream() {
  return stream_;
}

template < typename M >
template < typename T >
void Serializer< M >::save(const T &t) {
  if (dynamic_cast< BinarySerializer< M >* >(this) != nullptr) {
    dynamic_cast< BinarySerializer< M >* >(this)->save(t);
  } else if (dynamic_cast< TextSerializer< M >* >(this) != nullptr) {
    dynamic_cast< TextSerializer< M >* >(this)->save(t);
  }
}

template < typename M >
template < typename T >
void Serializer< M >::load(T *t) {
  if (dynamic_cast< BinarySerializer< M >* >(this) != nullptr) {
    dynamic_cast< BinarySerializer< M >* >(this)->load(t);
  } else if (dynamic_cast< TextSerializer< M >* >(this) != nullptr) {
    dynamic_cast< TextSerializer< M >* >(this)->load(t);
  }
}

}  // namespace serial
}  // namespace thunder

#endif  // THUNDER_SERIALIZER_SERIALIZER_INL_HPP_
