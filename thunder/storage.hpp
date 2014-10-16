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

#ifndef THUNDER_STORAGE_HPP
#define THUNDER_STORAGE_HPP

#include "thunder/storage/storage.hpp"

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

#endif  // THUNDER_STORAGE_HPP
