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

#ifndef THUNDER_SERIALIZER_HPP
#define THUNDER_SERIALIZER_HPP

#include <fstream>
#include <sstream>

#include "thunder/serializer/binary_protocol.hpp"
#include "thunder/serializer/serializer.hpp"
#include "thunder/serializer/static.hpp"
#include "thunder/serializer/text_protocol.hpp"

namespace thunder {

template < typename P = serializer::TextProtocol< ::std::stringstream > >
using Serializer = serializer::Serializer< P >;

template < typename M = ::std::stringstream >
using BinarySerializer = Serializer< serializer::BinaryProtocol< M > >;
typedef BinarySerializer< ::std::stringstream > StringBinarySerializer;
typedef BinarySerializer< ::std::fstream > FileBinarySerializer;

template < typename M = ::std::stringstream >
using TextSerializer = Serializer< serializer::TextProtocol< M > >;
typedef TextSerializer< ::std::stringstream > StringTextSerializer;
typedef TextSerializer< ::std::fstream > FileTextSerializer;

}  // namespace thunder

namespace thunder {
namespace serializer {

extern template class BinaryProtocol< ::std::stringstream >;
extern template class BinaryProtocol< ::std::fstream >;
extern template class TextProtocol< ::std::stringstream >;
extern template class TextProtocol< ::std::fstream >;

extern template class Serializer< BinaryProtocol< ::std::stringstream > >;
extern template class Serializer< BinaryProtocol< ::std::fstream > >;
extern template class Serializer< TextProtocol< ::std::stringstream > >;
extern template class Serializer< TextProtocol< ::std::fstream > >;

#define THUNDER_SERIALIZER_INSTANTIATE_STRINGSTREAM_CONSTRUCTOR(P)      \
  extern template Serializer< P >::Serializer();                        \
  extern template Serializer< P >::Serializer(openmode mode);           \
  extern template Serializer< P >::Serializer(const ::std::string &str); \
  extern template Serializer< P >::Serializer(                          \
      const ::std::string &str, openmode mode);

THUNDER_SERIALIZER_INSTANTIATE_STRINGSTREAM_CONSTRUCTOR(
    BinaryProtocol< ::std::stringstream >);
THUNDER_SERIALIZER_INSTANTIATE_STRINGSTREAM_CONSTRUCTOR(
    TextProtocol< ::std::stringstream >);

#undef THUNDER_SERIALIZER_INSTANTIATE_STRINGSTREAM_CONSTRUCTOR

#define THUNDER_SERIALIZER_INSTANTIATE_FSTREAM_CONSTRUCTOR(P)           \
  extern template Serializer< P >::Serializer();                        \
  extern template Serializer< P >::Serializer(const char * filename);   \
  extern template Serializer< P >::Serializer(                          \
      const char * filename, openmode mode);                            \
  extern template Serializer< P >::Serializer(const ::std::string &filename); \
  extern template Serializer< P >::Serializer(                          \
      const ::std::string &filename, openmode mode);

THUNDER_SERIALIZER_INSTANTIATE_FSTREAM_CONSTRUCTOR(
    BinaryProtocol< ::std::fstream >);
THUNDER_SERIALIZER_INSTANTIATE_FSTREAM_CONSTRUCTOR(
    TextProtocol< ::std::fstream >);

#undef THUNDER_SERIALIZER_INSTANTIATE_FSTREAM_CONSTRUCTOR

}  // namespace serializer
}  // namespace thunder

#endif  // THUNDER_SERIALIZER_HPP
