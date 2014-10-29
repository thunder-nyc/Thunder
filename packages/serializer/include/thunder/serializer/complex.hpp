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

#ifndef THUNDER_SERIALIZER_COMPLEX_HPP_
#define THUNDER_SERIALIZER_COMPLEX_HPP_

#include <complex>

#include "thunder/serializer/serializer.hpp"

namespace thunder {
namespace serializer {

template < typename S, typename T >
void save(S *s, const ::std::complex< T > &t);

template < typename S, typename T >
void load(S *s, ::std::complex< T > *t);

}  // namespace serializer
}  // namespace thunder

#endif  // THUNDER_SERIALIZER_COMPLEX_HPP_
