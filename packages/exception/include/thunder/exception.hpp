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

#ifndef THUNDER_EXCEPTION_HPP_
#define THUNDER_EXCEPTION_HPP_

#include "thunder/exception/exception.hpp"

namespace thunder {

using exception::logic_error;
using exception::runtime_error;
using exception::invalid_argument;
using exception::domain_error;
using exception::length_error;
using exception::out_of_range;
using exception::contiguity_error;
using exception::range_error;
using exception::overflow_error;
using exception::underflow_error;

}  // namespace thunder

#endif  // THUNDER_EXCEPTION_HPP_
