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

#include <fstream>
#include <ios>
#include <sstream>
#include <string>

#include "thunder/serializer/binary_protocol.hpp"
#include "thunder/serializer/serializer.hpp"
#include "thunder/serializer/static.hpp"
#include "thunder/serializer/text_protocol.hpp"

#include "thunder/serializer/binary_protocol-inl.hpp"
#include "thunder/serializer/serializer-inl.hpp"
#include "thunder/serializer/static-inl.hpp"
#include "thunder/serializer/text_protocol-inl.hpp"

namespace thunder {
namespace serializer {

template class BinaryProtocol< ::std::stringstream >;
template class BinaryProtocol< ::std::fstream >;
template class TextProtocol< ::std::stringstream >;
template class TextProtocol< ::std::fstream >;

template class Serializer< BinaryProtocol< ::std::stringstream > >;
template class Serializer< BinaryProtocol< ::std::fstream > >;
template class Serializer< TextProtocol< ::std::stringstream > >;
template class Serializer< TextProtocol< ::std::fstream > >;

// The non-const non-reference string constructors have to be done because
// the linker will try to find it even if there is a const reference parameter
// This is the case with LLVM/Clang 7.0 and GCC 8.2.
// A better approach is to use ::std::enable_if with SFINAE.
#define THUNDER_SERIALIZER_INSTANTIATE_STRINGSTREAM_CONSTRUCTOR(P)      \
  template Serializer< P >::Serializer();                               \
  template Serializer< P >::Serializer(openmode mode);                  \
  template Serializer< P >::Serializer(const ::std::string &str);       \
  template Serializer< P >::Serializer(                                 \
      const ::std::string &str, openmode mode);                         \
  template Serializer< P >::Serializer(::std::string str);              \
  template Serializer< P >::Serializer(                                 \
      ::std::string str, openmode mode);

THUNDER_SERIALIZER_INSTANTIATE_STRINGSTREAM_CONSTRUCTOR(
    BinaryProtocol< ::std::stringstream >);
THUNDER_SERIALIZER_INSTANTIATE_STRINGSTREAM_CONSTRUCTOR(
    TextProtocol< ::std::stringstream >);

#undef THUNDER_SERIALIZER_INSTANTIATE_STRINGSTREAM_CONSTRUCTOR

#define THUNDER_SERIALIZER_INSTANTIATE_FSTREAM_CONSTRUCTOR(P)           \
  template Serializer< P >::Serializer();                               \
  template Serializer< P >::Serializer(const char * filename);          \
  template Serializer< P >::Serializer(const char * filename, openmode mode); \
  template Serializer< P >::Serializer(const ::std::string &filename);  \
  template Serializer< P >::Serializer(                                 \
      const ::std::string &filename, openmode mode);                    \
  template Serializer< P >::Serializer(::std::string filename);         \
  template Serializer< P >::Serializer(                                 \
      ::std::string filename, openmode mode);

THUNDER_SERIALIZER_INSTANTIATE_FSTREAM_CONSTRUCTOR(
    BinaryProtocol< ::std::fstream >);
THUNDER_SERIALIZER_INSTANTIATE_FSTREAM_CONSTRUCTOR(
    TextProtocol< ::std::fstream >);

#undef THUNDER_SERIALIZER_INSTANTIATE_FSTREAM_CONSTRUCTOR

#define THUNDER_SERIALIZER_INSTANTIATE_BASIC_SERIALIZE(P)              \
  template void Serializer< P >::save(const char &t);                  \
  template void Serializer< P >::load(char *t);                        \
  template void Serializer< P >::save(const signed char &t);           \
  template void Serializer< P >::load(signed char *t);                 \
  template void Serializer< P >::save(const unsigned char &t);         \
  template void Serializer< P >::load(unsigned char *t);               \
  template void Serializer< P >::save(const wchar_t &t);               \
  template void Serializer< P >::load(wchar_t *t);                     \
  template void Serializer< P >::save(const char16_t &t);              \
  template void Serializer< P >::load(char16_t *t);                    \
  template void Serializer< P >::save(const char32_t &t);              \
  template void Serializer< P >::load(char32_t *t);                    \
  template void Serializer< P >::save(const short &t);                 \
  template void Serializer< P >::load(short *t);                       \
  template void Serializer< P >::save(const unsigned short &t);        \
  template void Serializer< P >::load(unsigned short *t);              \
  template void Serializer< P >::save(const int &t);                   \
  template void Serializer< P >::load(int *t);                         \
  template void Serializer< P >::save(const unsigned int &t);          \
  template void Serializer< P >::load(unsigned int *t);                \
  template void Serializer< P >::save(const long &t);                  \
  template void Serializer< P >::load(long *t);                        \
  template void Serializer< P >::save(const unsigned long &t);         \
  template void Serializer< P >::load(unsigned long *t);               \
  template void Serializer< P >::save(const long long &t);             \
  template void Serializer< P >::load(long long *t);                   \
  template void Serializer< P >::save(const unsigned long long &t);    \
  template void Serializer< P >::load(unsigned long long *t);          \
  template void Serializer< P >::save(const float &t);                 \
  template void Serializer< P >::load(float *t);                       \
  template void Serializer< P >::save(const double &t);                \
  template void Serializer< P >::load(double *t);                      \
  template void Serializer< P >::save(const long double &t);           \
  template void Serializer< P >::load(long double *t);

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
