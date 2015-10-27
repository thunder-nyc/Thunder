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


#include "thunder/linalg/linalg.hpp"

#include "thunder/linalg/math.hpp"
#include "thunder/tensor.hpp"

#include "thunder/linalg/linalg-inl.hpp"
#include "thunder/linalg/math-inl.hpp"

namespace thunder {
namespace linalg {

template class Linalg< DoubleTensor >;
template class Linalg< FloatTensor >;
template class Linalg< DoubleComplexTensor >;
template class Linalg< FloatComplexTensor >;

}  // namespace linalg
}  // namespace thunder
