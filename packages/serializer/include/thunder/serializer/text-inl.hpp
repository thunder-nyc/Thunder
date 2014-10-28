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

#ifndef THUNDER_SERIALIZER_TEXT_INL_HPP_
#define THUNDER_SERIALIZER_TEXT_INL_HPP_

#include <sstream>

#include "thunder/serializer/serializer.hpp"

namespace thunder {
namespace serializer {

template < typename M >
template < typename S, typename T >
void Text< M >::save(S *s, const T &t) {
  ::thunder::serializer::save(s, t);
}

template < typename M >
template < typename S, typename T >
void Text< M >::load(S *s, T *t) {
  ::thunder::serializer::load(s, t);
}

}
}

#endif // THUNDER_SERIALIZER_TEXT_INL_HPP_
