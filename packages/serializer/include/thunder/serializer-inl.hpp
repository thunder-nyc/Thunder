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

#ifndef THUNDER_SERIALIZER_INL_HPP
#define THUNDER_SERIALIZER_INL_HPP

#include <fstream>
#include <sstream>

#include "thunder/serializer.hpp"
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

extern template class BinaryProtocol< ::std::stringstream >;
extern template class BinaryProtocol< ::std::fstream >;
extern template class TextProtocol< ::std::stringstream >;
extern template class TextProtocol< ::std::fstream >;

extern template class Serializer< BinaryProtocol< ::std::stringstream > >;
extern template class Serializer< BinaryProtocol< ::std::fstream > >;
extern template class Serializer< TextProtocol< ::std::stringstream > >;
extern template class Serializer< TextProtocol< ::std::fstream > >;

}  // namespace serializer
}  // namespace thunder

#endif  // THUNDER_SERIALIZER_INL_HPP
