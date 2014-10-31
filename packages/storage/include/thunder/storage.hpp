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

#ifndef THUNDER_STORAGE_HPP_
#define THUNDER_STORAGE_HPP_

#include "thunder/storage/storage.hpp"
#include "thunder/serializer.hpp"

#include <memory>
#include <complex>
#include <utility>

namespace thunder {

template < typename D = double, typename A = ::std::allocator< D > >
using Storage = storage::Storage< D, A >;
typedef Storage< double, ::std::allocator< double > > DoubleStorage;
typedef Storage< float, ::std::allocator< float > > FloatStorage;
typedef Storage< ::std::size_t, ::std::allocator< ::std::size_t > > SizeStorage;

template < typename D = double,
           typename A = ::std::allocator< ::std::complex< D > > >
using ComplexStorage = storage::Storage< ::std::complex< D >, A >;
typedef ComplexStorage< double, ::std::allocator< ::std::complex< double > > >
DoubleComplexStorage;
typedef ComplexStorage< float, ::std::allocator< ::std::complex< float > > >
FloatComplexStorage;

}  // namespace thunder

namespace thunder {
namespace storage {

extern template class Storage< double >;
extern template class Storage< float >;
extern template class Storage< ::std::complex< double > >;
extern template class Storage< ::std::complex< float > >;
extern template class Storage< ::std::size_t >;
extern template class Storage< ::std::ptrdiff_t >;
extern template class Storage< ::std::pair< ::std::size_t, ::std::size_t > >;

}  // namespace storage
}  // namespace thunder

namespace thunder {
namespace serializer {

#define THUNDER_STORAGE_INSTANTIATE_SERIALIZE(D)                        \
  extern template void StringBinarySerializer::save(                    \
      const ::thunder::storage::Storage< D > &t);                       \
  extern template void StringBinarySerializer::load(                    \
      ::thunder::storage::Storage< D > *t);                             \
  extern template void FileBinarySerializer::save(                      \
      const ::thunder::storage::Storage< D > &t);                       \
  extern template void FileBinarySerializer::load(                      \
      ::thunder::storage::Storage< D > *t);                             \
  extern template void StringTextSerializer::save(                      \
      const ::thunder::storage::Storage< D > &t);                       \
  extern template void StringTextSerializer::load(                      \
      ::thunder::storage::Storage< D > *t);                             \
  extern template void FileTextSerializer::save(                        \
      const ::thunder::storage::Storage< D > &t);                       \
  extern template void FileTextSerializer::load(                        \
      ::thunder::storage::Storage< D > *t);

#define THUNDER_STORAGE_EXPAND_SERIALIZE(INSTANTIATE)           \
  INSTANTIATE(double);                                          \
  INSTANTIATE(float);                                           \
  INSTANTIATE(::std::complex< double >);                        \
  INSTANTIATE(::std::complex< float >);                         \
  INSTANTIATE(::std::size_t);                                   \
  INSTANTIATE(::std::ptrdiff_t);

extern template void StringBinarySerializer::save(
    const ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > &t);
extern template void StringBinarySerializer::load(
    ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > *t);
extern template void FileBinarySerializer::save(
    const ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > &t);
extern template void FileBinarySerializer::load(
    ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > *t);
extern template void StringTextSerializer::save(
    const ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > &t);
extern template void StringTextSerializer::load(
    ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > *t);
extern template void FileTextSerializer::save(
    const ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > &t);
extern template void FileTextSerializer::load(
    ::thunder::storage::Storage<
    ::std::pair< ::std::size_t, ::std::size_t > > *t);

THUNDER_STORAGE_EXPAND_SERIALIZE(THUNDER_STORAGE_INSTANTIATE_SERIALIZE);

#undef THUNDER_STORAGE_INSTANTIATE_SERIALIZE
#undef THUNDER_STORAGE_EXPAND_SERIALIZE

}  // namespace serializer
}  // namespace thunder

#endif  // THUNDER_STORAGE_HPP_
