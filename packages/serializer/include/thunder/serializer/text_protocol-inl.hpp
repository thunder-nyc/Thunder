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

#ifndef THUNDER_SERIALIZER_TEXT_PROTOCOL_INL_HPP_
#define THUNDER_SERIALIZER_TEXT_PROTOCOL_INL_HPP_

#include "thunder/serializer/text_protocol.hpp"

#include <iomanip>
#include <limits>

#include "thunder/serializer/serializer.hpp"
#include "thunder/serializer/static.hpp"

namespace thunder {
namespace serializer {

template < typename M >
template < typename... G >
TextProtocol< M >::TextProtocol(G... g) : stream_(g...) {}

template < typename M >
const typename TextProtocol< M >::stream_type&
TextProtocol< M >::stream() const {
  return stream_;
}

template < typename M >
typename TextProtocol< M >::stream_type& TextProtocol< M >::stream() {
  return const_cast< stream_type& >(
      const_cast< const TextProtocol* >(this)->stream());
}

template < typename M >
template < typename S, typename T >
void TextProtocol< M >::save(S *s, const T &t) {
  ::thunder::serializer::save(s, t);
}

template < typename M >
template < typename S, typename T >
void TextProtocol< M >::load(S *s, T *t) {
  ::thunder::serializer::load(s, t);
}

#define THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_CHAR(CHAR_TYPE, INT_TYPE) \
  template < typename M >                                               \
  template < typename S >                                               \
  void TextProtocol< M >::save(S *s, const CHAR_TYPE &t) {              \
    save(s, static_cast< INT_TYPE >(t));                                \
  }                                                                     \
  template < typename M >                                               \
  template < typename S >                                               \
  void TextProtocol< M >::load(S *s, CHAR_TYPE *t) {                    \
    INT_TYPE t_val;                                                     \
    load(s, &t_val);                                                    \
    *t = static_cast< CHAR_TYPE >(t_val);                               \
  }

THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_CHAR(char, short);
THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_CHAR(signed char, short);
THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_CHAR(unsigned char, unsigned short);
THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_CHAR(wchar_t, long long);
THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_CHAR(char16_t, short);
THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_CHAR(char32_t, long);

#undef THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_CHAR

#define THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_INTEGER(TYPE)           \
  template < typename M >                                               \
  template < typename S >                                               \
  void TextProtocol< M >::save(S *s, const TYPE &t) {                   \
    stream_ << t << ' ';                                                \
  }                                                                     \
  template < typename M >                                               \
  template < typename S >                                               \
  void TextProtocol< M >::load(S *s, TYPE *t) {                         \
    stream_ >> (*t);                                                    \
  }

THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_INTEGER(short);
THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_INTEGER(unsigned short);
THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_INTEGER(int);
THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_INTEGER(unsigned int);
THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_INTEGER(long);
THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_INTEGER(unsigned long);
THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_INTEGER(long long);
THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_INTEGER(unsigned long long);

#undef THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_INTEGER

#define THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_FLOAT(TYPE)             \
  template < typename M >                                               \
  template < typename S >                                               \
  void TextProtocol< M >::save(S *s, const TYPE &t) {                   \
    stream_ << ::std::scientific << ::std::setprecision(                \
        ::std::numeric_limits< TYPE >::max_digits10) << t << ' ';       \
  }                                                                     \
  template < typename M >                                               \
  template < typename S >                                               \
  void TextProtocol< M >::load(S *s, TYPE *t) {                         \
    stream_ >> (*t);                                                    \
  }

THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_FLOAT(float);
THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_FLOAT(double);
THUNDER_SERIALIZER_TEXT_PROTOCOL_DEFINE_FLOAT(long double);

#undef THUNDER_SERIALIZATION_TEXT_PROTOCOL_DEFINE_FLOAT

}  // namespace serializer
}  // namespace thunder

#endif  // THUNDER_SERIALIZER_TEXT_PROTOCOL_INL_HPP_
