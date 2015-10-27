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

#ifndef THUNDER_LINALG_HPP_
#define THUNDER_LINALG_HPP_

#include "thunder/linalg/linalg.hpp"

#include "thunder/linalg/type.hpp"
#include "thunder/tensor.hpp"

namespace thunder {

template < typename T, typename H = typename linalg::type< T >::handle_type >
using Linalg = linalg::Linalg< T, H >;

typedef Linalg< DoubleTensor > DoubleLinalg;
typedef Linalg< FloatTensor > FloatLinalg;
typedef Linalg< DoubleComplexTensor > DoubleComplexLinalg;
typedef Linalg< FloatComplexTensor > FloatComplexLinalg;

}

namespace thunder {
namespace linalg {

extern template class Linalg< DoubleTensor >;
extern template class Linalg< FloatTensor >;
extern template class Linalg< DoubleComplexTensor >;
extern template class Linalg< FloatComplexTensor >;

}  // namespace linalg
}  // namespace thunder

#endif  // THUNDER_LINALG_HPP_
