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
#include <string>
#include <unordered_map>

#include "thunder/serializer/text_protocol.hpp"

namespace thunder {
namespace serializer {

template < typename P = TextProtocol< ::std::stringstream > >
class Serializer {
 public:
  typedef P protocol_type;
  typedef typename P::stream_type stream_type;
  typedef typename P::openmode openmode;

  // Introducing openmode constants
  static constexpr openmode app = P::app;
  static constexpr openmode binary = P::binary;
  static constexpr openmode in = P::in;
  static constexpr openmode out = P::out;
  static constexpr openmode trunc = P::trunc;
  static constexpr openmode ate = P::ate;

  template < typename... G >
  explicit Serializer(G... g);

  // Disable copy constructor and assignment
  Serializer(const Serializer &s) = delete;
  Serializer& operator=(Serializer s) = delete;

  const protocol_type& protocol() const;

  template < typename T >
  void save(const T &t);
  template < typename T >
  void load(T *t);

  // Pointer save: record the saved pointer
  template < typename T >
  void save(T* const &t);
  // Pointer load: record the loaded pointer
  template < typename T >
  void load(T* *t);

  // shared_pointer save: similar to pointer save
  template < typename T >
  void save(const ::std::shared_ptr< T > &t);
  // shared_pointer load: similar to pointer load
  template < typename T >
  void load(::std::shared_ptr< T > *t);

 private:
  P protocol_;
  unsigned int saved_count_;
  ::std::unordered_map< void*, unsigned int > saved_pointers_;
  ::std::unordered_map< unsigned int, void* > loaded_pointers_;
  ::std::unordered_map< unsigned int, void* > loaded_shared_;
};

}  // namespace serializer
}  // namespace thunder

#endif  // THUNDER_SERIALIZER_SERIALIZER_HPP_
