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

#ifndef THUNDER_SERIALIZER_TEXT_HPP_
#define THUNDER_SERIALIZER_TEXT_HPP_

#include <sstream>

namespace thunder {
namespace serializer {

template < typename M = ::std::stringstream >
class Text {
 public:
  typedef M stream_type;
  typedef ::std::shared_ptr< M > stream_pointer;

  template < typename... G >
  Text(G... g);

  stream_pointer stream() const;

  // Listing of fundamental C++11 character type serialization
  void save(const char &t);
  void load(char *t);
  void save(const unsigned char &t);
  void load(unsigned char *t);
  void save(const wchar_t &t);
  void load(wchar_t *t);
  void save(const char16_t &t);
  void load(char16_t *t);
  void save(const char32_t &t);
  void load(char32_t *t);

  // Listing of fundamental C++11 integer type serialization
  void save(const short &t);
  void load(short *t);
  void save(const unsigned short &t);
  void load(unsigned short *t);
  void save(const int &t);
  void load(int *t);
  void save(const unsigned int &t);
  void load(unsigned int &t);
  void save(const long &t);
  void load(long *t);
  void save(const long long &t);
  void load(long long *t);

  // Listing of fundamental C++11 floating point type serialization
  void save(const float &t);
  void load(float *t);
  void save(const double &t);
  void load(double *t);
  void save(const long double &t);
  void load(long double *t);

  void save(const void* &t);
  void load(void* *t);

 private:
  stream_pointer stream_;
};

}  // namespace serializer
}  // namespace thunder

#endif  // THUNDER_SERIALIZER_TEXT_HPP_
