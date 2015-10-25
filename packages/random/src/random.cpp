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

#include "thunder/random/random.hpp"

#include <ctime>
#include <random>

#include "thunder/random/math.hpp"
#include "thunder/tensor.hpp"

#include "thunder/random/math-inl.hpp"
#include "thunder/random/random-inl.hpp"

namespace thunder {
namespace random {

#define THUNDER_RANDOM_INSTANTIATE(T)                                   \
  template class Random< T >;                                           \
  template Random< T >::Random();                                       \
  template Random< T >::Random(typename ::std::mt19937::result_type val); \
  template Random< T >::Random(int val);                                \
  template Random< T >::Random(::std::time_t val);                      \
  template Random< T >::Random(::std::seed_seq q);

// int and double are default types
THUNDER_RANDOM_INSTANTIATE(DoubleTensor);
THUNDER_RANDOM_INSTANTIATE(FloatTensor);
THUNDER_RANDOM_INSTANTIATE(SizeTensor);

#undef THUNDR_RANDOM_INSTANTIATE

}  // namespace random
}  // namespace thunder
