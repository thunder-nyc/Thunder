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

#ifndef THUNDER_EXCEPTION_CONTIGUITY_ERROR_HPP_
#define THUNDER_EXCEPTION_CONTIGUITY_ERROR_HPP_

#include <string>

#include "thunder/exception/exception.hpp"

namespace thunder {
namespace exception {

class contiguity_error : public runtime_error {
 public:
  explicit contiguity_error(const std::string& what_arg);
  explicit contiguity_error(const char* what_arg);
};

}  // namespace exception
}  // namespace thunder

#endif  // THUNDER_EXCEPTION_CONTIGUITY_ERROR_HPP_
