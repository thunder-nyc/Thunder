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

#ifndef THUNDER_TENSOR_MATH_INL_STATIC_HPP_
#define THUNDER_TENSOR_MATH_INL_STATIC_HPP_

#include "thunder/tensor/math.hpp"
#include "thunder/tensor/math-inl.hpp"

#include <cmath>
#include <complex>

#include "thunder/exception.hpp"

namespace thunder {
namespace tensor {
namespace math {

template < typename T1, typename T2 >
T1 polar(typename T2::const_reference r, const T2& theta) {
  throw domain_error("polar is undefined for real numbers");
  return T1();
}

template < typename T1, typename T2 >
T1 polar(const T2& r, typename T2::const_reference theta) {
  throw domain_error("polar is undefined for real numbers");
  return T1();
}

template < typename T1, typename T2 >
T1 polar(const T2& r, const T2& theta) {
  throw domain_error("polar is undefined for real numbers");
  return T1();
}

}  // namespace math
}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_MATH_INL_STATIC_HPP_
