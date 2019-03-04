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
      const ::std::string &str, openmode mode);                         \
  extern template Serializer< P >::Serializer(::std::string str);       \
  extern template Serializer< P >::Serializer(                          \
      ::std::string str, openmode mode);

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
      const ::std::string &filename, openmode mode);                    \
  extern template Serializer< P >::Serializer(::std::string filename);  \
  extern template Serializer< P >::Serializer(                          \
      ::std::string filename, openmode mode);

THUNDER_SERIALIZER_INSTANTIATE_FSTREAM_CONSTRUCTOR(
    BinaryProtocol< ::std::fstream >);
THUNDER_SERIALIZER_INSTANTIATE_FSTREAM_CONSTRUCTOR(
    TextProtocol< ::std::fstream >);

#undef THUNDER_SERIALIZER_INSTANTIATE_FSTREAM_CONSTRUCTOR

#define THUNDER_SERIALIZER_INSTANTIATE_BASIC_SERIALIZE(P)               \
  extern template void Serializer< P >::save(const char &t);            \
  extern template void Serializer< P >::load(char *t);                  \
  extern template void Serializer< P >::save(const signed char &t);     \
  extern template void Serializer< P >::load(signed char *t);           \
  extern template void Serializer< P >::save(const unsigned char &t);   \
  extern template void Serializer< P >::load(unsigned char *t);         \
  extern template void Serializer< P >::save(const wchar_t &t);         \
  extern template void Serializer< P >::load(wchar_t *t);               \
  extern template void Serializer< P >::save(const char16_t &t);        \
  extern template void Serializer< P >::load(char16_t *t);              \
  extern template void Serializer< P >::save(const char32_t &t);        \
  extern template void Serializer< P >::load(char32_t *t);              \
  extern template void Serializer< P >::save(const short &t);           \
  extern template void Serializer< P >::load(short *t);                 \
  extern template void Serializer< P >::save(const unsigned short &t);  \
  extern template void Serializer< P >::load(unsigned short *t);        \
  extern template void Serializer< P >::save(const int &t);             \
  extern template void Serializer< P >::load(int *t);                   \
  extern template void Serializer< P >::save(const unsigned int &t);    \
  extern template void Serializer< P >::load(unsigned int *t);          \
  extern template void Serializer< P >::save(const long &t);            \
  extern template void Serializer< P >::load(long *t);                  \
  extern template void Serializer< P >::save(const unsigned long &t);   \
  extern template void Serializer< P >::load(unsigned long *t);         \
  extern template void Serializer< P >::save(const long long &t);       \
  extern template void Serializer< P >::load(long long *t);             \
  extern template void Serializer< P >::save(const unsigned long long &t); \
  extern template void Serializer< P >::load(unsigned long long *t);    \
  extern template void Serializer< P >::save(const float &t);           \
  extern template void Serializer< P >::load(float *t);                 \
  extern template void Serializer< P >::save(const double &t);          \
  extern template void Serializer< P >::load(double *t);                \
  extern template void Serializer< P >::save(const long double &t);     \
  extern template void Serializer< P >::load(long double *t);

THUNDER_SERIALIZER_INSTANTIATE_BASIC_SERIALIZE(
    BinaryProtocol< ::std::stringstream >);
THUNDER_SERIALIZER_INSTANTIATE_BASIC_SERIALIZE(
    BinaryProtocol< ::std::fstream >);
THUNDER_SERIALIZER_INSTANTIATE_BASIC_SERIALIZE(
    TextProtocol< ::std::stringstream >);
THUNDER_SERIALIZER_INSTANTIATE_BASIC_SERIALIZE(
    TextProtocol< ::std::fstream >);

#undef THUNDER_SERIALIZER_INSTANTIATE_BASIC_SERIALIZE

}  // namespace serializer
}  // namespace thunder

#endif  // THUNDER_SERIALIZER_HPP
