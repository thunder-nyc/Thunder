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

#include "thunder/storage/storage.hpp"

#include <complex>
#include <utility>

#include "thunder/serializer.hpp"
#include "thunder/serializer/binary_protocol.hpp"
#include "thunder/serializer/serializer.hpp"
#include "thunder/serializer/static.hpp"
#include "thunder/serializer/text_protocol.hpp"

#include "thunder/serializer/binary_protocol-inl.hpp"
#include "thunder/serializer/serializer-inl.hpp"
#include "thunder/serializer/static-inl.hpp"
#include "thunder/serializer/text_protocol-inl.hpp"
#include "thunder/storage/storage-inl.hpp"

namespace thunder {
namespace storage {

template class Storage< double >;
template class Storage< float >;
template class Storage< ::std::complex< double > >;
template class Storage< ::std::complex< float > >;
template class Storage< ::std::size_t >;
template class Storage< ::std::ptrdiff_t >;
template class Storage< ::std::pair< ::std::size_t, ::std::size_t > >;

template Storage< double > Storage<
  ::std::complex< double > >::view< Storage< double > >();
template Storage< float > Storage<
  ::std::complex< float > >::view< Storage< float > >();

}  // namespace storage
}  // namespace thunder

namespace thunder {
namespace serializer {

#define THUNDER_STORAGE_INSTANTIATE_SERIALIZE(D)                \
  template void save(                                           \
      StringBinarySerializer *s,                                \
      const ::thunder::storage::Storage< D > &t);               \
  template void StringBinarySerializer::save(                   \
      const ::thunder::storage::Storage< D > &t);               \
  template void load(                                           \
      StringBinarySerializer *s,                                \
      ::thunder::storage::Storage< D > *t);                     \
  template void StringBinarySerializer::load(                   \
      ::thunder::storage::Storage< D > *t);                     \
  template void save(                                           \
      FileBinarySerializer *s,                                  \
      const ::thunder::storage::Storage< D > &t);               \
  template void FileBinarySerializer::save(                     \
      const ::thunder::storage::Storage< D > &t);               \
  template void load(                                           \
      FileBinarySerializer *s,                                  \
      ::thunder::storage::Storage< D > *t);                     \
  template void FileBinarySerializer::load(                     \
      ::thunder::storage::Storage< D > *t);                     \
  template void save(                                           \
      StringTextSerializer *s,                                  \
      const ::thunder::storage::Storage< D > &t);               \
  template void StringTextSerializer::save(                     \
      const ::thunder::storage::Storage< D > &t);               \
  template void load(                                           \
      StringTextSerializer *s,                                  \
      ::thunder::storage::Storage< D > *t);                     \
  template void StringTextSerializer::load(                     \
      ::thunder::storage::Storage< D > *t);                     \
  template void save(                                           \
      FileTextSerializer *s,                                    \
      const ::thunder::storage::Storage< D > &t);               \
  template void FileTextSerializer::save(                       \
      const ::thunder::storage::Storage< D > &t);               \
  template void load(                                           \
      FileTextSerializer *s,                                    \
      ::thunder::storage::Storage< D > *t);                     \
  template void FileTextSerializer::load(                       \
      ::thunder::storage::Storage< D > *t);


#define THUNDER_STORAGE_EXPAND_SERIALIZE(INSTANTIATE)           \
  INSTANTIATE(double);                                          \
  INSTANTIATE(float);                                           \
  INSTANTIATE(::std::complex< double >);                        \
  INSTANTIATE(::std::complex< float >);                         \
  INSTANTIATE(::std::size_t);                                   \
  INSTANTIATE(::std::ptrdiff_t);

template void save(
    StringBinarySerializer *s,
    const ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > &t);
template void StringBinarySerializer::save(
    const ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > &t);
template void load(
    StringBinarySerializer *s,
    ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > *t);
template void StringBinarySerializer::load(
    ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > *t);
template void save(
    FileBinarySerializer *s,
    const ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > &t);
template void FileBinarySerializer::save(
    const ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > &t);
template void load(
    FileBinarySerializer *s,
    ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > *t);
template void FileBinarySerializer::load(
    ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > *t);
template void save(
    StringTextSerializer *s,
    const ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > &t);
template void StringTextSerializer::save(
    const ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > &t);
template void load(
    StringTextSerializer *s,
    ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > *t);
template void StringTextSerializer::load(
    ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > *t);
template void save(
    FileTextSerializer *s,
    const ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > &t);
template void FileTextSerializer::save(
    const ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > &t);
template void load(
    FileTextSerializer *s,
    ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > *t);
template void FileTextSerializer::load(
    ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > *t);

THUNDER_STORAGE_EXPAND_SERIALIZE(THUNDER_STORAGE_INSTANTIATE_SERIALIZE);

#undef THUNDER_STORAGE_INSTANTIATE_SERIALIZE
#undef THUNDER_STORAGE_EXPAND_SERIALIZE

}  // namespace serializer
}  // namespace thunder
