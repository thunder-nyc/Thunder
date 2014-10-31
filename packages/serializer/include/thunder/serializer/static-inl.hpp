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

#ifndef THUNDER_SERIALIZER_STATIC_INL_HPP_
#define THUNDER_SERIALIZER_STATIC_INL_HPP_

#include "thunder/serializer/static.hpp"

#include <complex>

namespace thunder {
namespace serializer {

template < typename S, typename T >
void save(S *s, const T &t) {
  t.save(s);
}

template < typename S, typename T >
void load(S *s, T *t) {
  t->load(s);
}

template < typename S, typename T >
void save(S *s, const ::std::complex< T > &t) {
  s->save(t.real());
  s->save(t.imag());
}

template < typename S, typename T >
void load(S *s, ::std::complex< T > *t) {
  T real;
  s->load(&real);
  T imag;
  s->load(&imag);
  *t = ::std::complex< T >(real, imag);
}

template < typename S, typename T1, typename T2 >
void save(S *s, const ::std::pair< T1, T2 > &t) {
  s->save(t.first);
  s->save(t.second);
}

template < typename S, typename T1, typename T2 >
void load(S *s, ::std::pair< T1, T2 > *t) {
  T1 first;
  s->load(&first);
  T2 second;
  s->load(&second);
  *t = ::std::pair< T1, T2 >(first, second);
}

}  // namespace serializer
}  // namespace thunder

#endif  // THUNDER_SERIALIZER_STATIC_INL_HPP_
