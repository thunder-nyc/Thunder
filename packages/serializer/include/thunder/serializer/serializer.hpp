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

#ifndef THUNDER_SERIALIZER_SERIALIZER_HPP_
#define THUNDER_SERIALIZER_SERIALIZER_HPP_

#include <memory>
#include <sstream>

#include "thunder/serializer/text.hpp"

namespace thunder {
namespace serializer {

template < typename M = ::std::stringstream, typename P = Text< M > >
class Serializer {
 public:
  typedef M stream_type;
  typedef typename M::char_type char_type;
  typedef ::std::shared_ptr(M) stream_pointer;

  template < typename... G >
  Serializer(G... g);

  ~Serializer();

  stream_pointer stream();

  template < typename T >
  void save(const T &t);
  template < typename T >
  void load (T *t);

 private:
  stream_pointer *stream_;
  ::std::unordered_set< void* > saved_pointers_;
  ::std::unordered_map< void*, void* > loaded_pointers_;
  ::std::unordered_map< void*, void* > loaded_shared_;

  // Const variables to identify types data
  static const char_type kPrimitive = 0;
  static const char_type kClass = 1;
  static const char_type kPointer = 2;
  static const char_type kShared = 3;
};

}  // namespace serializer
}  // namespace thunder

#endif  // THUNDER_SERIALIZER_SERIALIZER_HPP_
