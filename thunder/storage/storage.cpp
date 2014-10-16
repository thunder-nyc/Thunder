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
#include "thunder/storage/storage-inl.hpp"

#include <complex>
#include <utility>

#include "boost/archive/binary_oarchive.hpp"
#include "boost/archive/binary_iarchive.hpp"
#include "boost/archive/text_oarchive.hpp"
#include "boost/archive/text_iarchive.hpp"
#include "boost/serialization/complex.hpp"
#include "boost/serialization/utility.hpp"

namespace thunder {
namespace storage {

template class Storage< double >;
template class Storage< float >;
template class Storage< ::std::complex< double > >;
template class Storage< ::std::complex< float > >;
template class Storage< ::std::size_t >;
template class Storage< ::std::ptrdiff_t >;
template class Storage< ::std::pair< ::std::size_t, ::std::size_t > >;

#define THUNDER_STORAGE_INSTANTIATE_SERIALIZE(D)                        \
  template void Storage< D >::serialize(                                \
      ::boost::archive::text_oarchive &ar, const unsigned int version); \
  template void Storage< D >::serialize(                                \
      ::boost::archive::text_iarchive &ar, const unsigned int version); \
  template void Storage< D >::serialize(                                \
      ::boost::archive::binary_oarchive &ar, const unsigned int version); \
  template void Storage< D >::serialize(                                \
      ::boost::archive::binary_iarchive &ar, const unsigned int version);

#define THUNDER_STORAGE_EXPAND_SERIALIZE(INSTANTIATE)           \
  INSTANTIATE(double);                                          \
  INSTANTIATE(float);                                           \
  INSTANTIATE(::std::complex< double >);                        \
  INSTANTIATE(::std::complex< float >);                         \
  INSTANTIATE(::std::size_t);                                   \
  INSTANTIATE(::std::ptrdiff_t);                                \

template void Storage< ::std::pair< ::std::size_t, ::std::size_t > >::serialize(
    ::boost::archive::text_oarchive &ar, const unsigned int version);
template void Storage< ::std::pair< ::std::size_t, ::std::size_t > >::serialize(
    ::boost::archive::text_iarchive &ar, const unsigned int version);
template void Storage< ::std::pair< ::std::size_t, ::std::size_t > >::serialize(
    ::boost::archive::binary_oarchive &ar, const unsigned int version);
template void Storage< ::std::pair< ::std::size_t, ::std::size_t > >::serialize(
    ::boost::archive::binary_iarchive &ar, const unsigned int version);

THUNDER_STORAGE_EXPAND_SERIALIZE(THUNDER_STORAGE_INSTANTIATE_SERIALIZE);

#undef THUNDER_STORAGE_INSTANTIATE_SERIALIZE
#undef THUNDER_STORAGE_EXPAND_SERIALIZE

}  // namespace storage
}  // namespace thunder
