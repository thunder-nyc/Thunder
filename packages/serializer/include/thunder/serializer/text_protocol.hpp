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

#ifndef THUNDER_SERIALIZER_TEXT_PROTOCOL_HPP_
#define THUNDER_SERIALIZER_TEXT_PROTOCOL_HPP_

#include <sstream>

namespace thunder {
namespace serializer {

template < typename M = ::std::stringstream >
class TextProtocol {
 public:
  typedef M stream_type;
  typedef typename M::openmode openmode;

  // Introducing openmode constants
  static constexpr openmode app = M::app;
  static constexpr openmode binary = M::binary;
  static constexpr openmode in = M::in;
  static constexpr openmode out = M::out;
  static constexpr openmode trunc = M::trunc;
  static constexpr openmode ate = M::ate;

  template < typename... G >
  explicit TextProtocol(G... g);

  // Disable copy constructor and assignment operator
  TextProtocol(const TextProtocol& p) = delete;
  TextProtocol& operator=(TextProtocol p) = delete;

  const stream_type& stream() const;
  stream_type& stream();

  // Save of generic type will call static methods
  template < typename S, typename T >
  void save(S *s, const T &t);
  template < typename S, typename T >
  void load(S *s, T *t);

  // Listing of fundamental C++11 character type serialization
  template < typename S >
  void save(S *s, const char &t);
  template < typename S >
  void load(S *s, char *t);
  template < typename S >
  void save(S *s, const signed char &t);
  template < typename S >
  void load(S *s, signed char *t);
  template < typename S >
  void save(S *s, const unsigned char &t);
  template < typename S >
  void load(S *s, unsigned char *t);
  template < typename S >
  void save(S *s, const wchar_t &t);
  template < typename S >
  void load(S *s, wchar_t *t);
  template < typename S >
  void save(S *s, const char16_t &t);
  template < typename S >
  void load(S *s, char16_t *t);
  template < typename S >
  void save(S *s, const char32_t &t);
  template < typename S >
  void load(S *s, char32_t *t);

  // Listing of fundamental C++11 integer type serialization
  template < typename S >
  void save(S *s, const short &t);
  template < typename S >
  void load(S *s, short *t);
  template < typename S >
  void save(S *s, const unsigned short &t);
  template < typename S >
  void load(S *s, unsigned short *t);
  template < typename S >
  void save(S *s, const int &t);
  template < typename S >
  void load(S *s, int *t);
  template < typename S >
  void save(S *s, const unsigned int &t);
  template < typename S >
  void load(S *s, unsigned int *t);
  template < typename S >
  void save(S *s, const long &t);
  template < typename S >
  void load(S *s, long *t);
  template < typename S >
  void save(S *s, const unsigned long &t);
  template < typename S >
  void load(S *s, unsigned long *t);
  template < typename S >
  void save(S *s, const long long &t);
  template < typename S >
  void load(S *s, long long *t);
  template < typename S >
  void save(S *s, const unsigned long long &t);
  template < typename S >
  void load(S *s, unsigned long long *t);

  // Listing of fundamental C++11 floating point type serialization
  template < typename S >
  void save(S *s, const float &t);
  template < typename S >
  void load(S *s, float *t);
  template < typename S >
  void save(S *s, const double &t);
  template < typename S >
  void load(S *s, double *t);
  template < typename S >
  void save(S *s, const long double &t);
  template < typename S >
  void load(S *s, long double *t);

 private:
  stream_type stream_;
};

}  // namespace serializer
}  // namespace thunder

#endif  // THUNDER_SERIALIZER_TEXT_PROTOCOL_HPP_
