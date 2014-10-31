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

#ifndef THUNDER_SERIALIZER_BINARY_PROTOCOL_INL_HPP_
#define THUNDER_SERIALIZER_BINARY_PROTOCOL_INL_HPP_

#include "thunder/serializer/binary_protocol.hpp"

#include <cstring>
#include <iomanip>
#include <ios>
#include <limits>

#include "thunder/serializer/serializer.hpp"
#include "thunder/serializer/static.hpp"

namespace thunder {
namespace serializer {

template < typename M >
template < typename... G >
BinaryProtocol< M >::BinaryProtocol(G... g) : stream_(g...) {}

template < typename M >
const typename BinaryProtocol< M >::stream_type&
BinaryProtocol< M >::stream() const {
  return stream_;
}

template < typename M >
typename BinaryProtocol< M >::stream_type& BinaryProtocol< M >::stream() {
  return const_cast< stream_type& >(
      const_cast< const BinaryProtocol* >(this)->stream());
}

template < typename M >
template < typename S, typename T >
void BinaryProtocol< M >::save(S *s, const T &t) {
  ::thunder::serializer::save(s, t);
}

template < typename M >
template < typename S, typename T >
void BinaryProtocol< M >::load(S *s, T *t) {
  ::thunder::serializer::load(s, t);
}



#define THUNDER_SERIALIZER_BINARY_PROTOCOL_DEFINE(TYPE)                 \
  template < typename M >                                               \
  template < typename S >                                               \
  void BinaryProtocol< M >::save(S *s, const TYPE &t) {                 \
    ::std::streamsize size = sizeof(TYPE) / sizeof(char_type);          \
    stream_.write(reinterpret_cast< const char_type* >(&t), size);      \
    ::std::streamsize rest = sizeof(TYPE) % sizeof(char_type);          \
    if (rest != 0) {                                                    \
      char buffer[sizeof(char_type)];                                   \
      ::std::memcpy(                                                    \
           buffer, reinterpret_cast< const char_type* >(&t) + size, rest); \
      stream_.write(reinterpret_cast< const char_type* >(buffer), 1);   \
    }                                                                   \
  }                                                                     \
  template < typename M >                                               \
  template < typename S >                                               \
  void BinaryProtocol< M >::load(S *s, TYPE *t) {                       \
    ::std::streamsize size = sizeof(TYPE) / sizeof(char_type);          \
    stream_.read(reinterpret_cast< char_type* >(t), size);              \
    ::std::streamsize rest = sizeof(TYPE) % sizeof(char_type);          \
    if (rest != 0) {                                                    \
      char buffer[sizeof(char_type)];                                   \
      stream_.read(reinterpret_cast< char_type* >(buffer), 1);          \
      ::std::memcpy(                                                    \
           reinterpret_cast< char_type* >(t) + size, buffer, rest);     \
    }                                                                   \
  }

THUNDER_SERIALIZER_BINARY_PROTOCOL_DEFINE(char);
THUNDER_SERIALIZER_BINARY_PROTOCOL_DEFINE(signed char);
THUNDER_SERIALIZER_BINARY_PROTOCOL_DEFINE(unsigned char);
THUNDER_SERIALIZER_BINARY_PROTOCOL_DEFINE(wchar_t);
THUNDER_SERIALIZER_BINARY_PROTOCOL_DEFINE(char16_t);
THUNDER_SERIALIZER_BINARY_PROTOCOL_DEFINE(char32_t);
THUNDER_SERIALIZER_BINARY_PROTOCOL_DEFINE(short);
THUNDER_SERIALIZER_BINARY_PROTOCOL_DEFINE(unsigned short);
THUNDER_SERIALIZER_BINARY_PROTOCOL_DEFINE(int);
THUNDER_SERIALIZER_BINARY_PROTOCOL_DEFINE(unsigned int);
THUNDER_SERIALIZER_BINARY_PROTOCOL_DEFINE(long);
THUNDER_SERIALIZER_BINARY_PROTOCOL_DEFINE(unsigned long);
THUNDER_SERIALIZER_BINARY_PROTOCOL_DEFINE(long long);
THUNDER_SERIALIZER_BINARY_PROTOCOL_DEFINE(unsigned long long);
THUNDER_SERIALIZER_BINARY_PROTOCOL_DEFINE(float);
THUNDER_SERIALIZER_BINARY_PROTOCOL_DEFINE(double);
THUNDER_SERIALIZER_BINARY_PROTOCOL_DEFINE(long double);

#undef THUNDER_SERIALIZER_BINARY_PROTOCOL_DEFINE

}  // namespace serializer
}  // namespace thunder

#endif  // THUNDER_SERIALIZER_BINARY_PROTOCOL_INL_HPP_
