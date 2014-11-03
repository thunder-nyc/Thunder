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

#define THUNDER_RANDOM_INSTANTIATE_MT19937(T, I, F)                     \
  template class Random< T, ::std::mt19937, I, F >;                     \
  template Random< T, ::std::mt19937, I, F >::Random();                 \
  template Random< T, ::std::mt19937, I, F >::                          \
  Random(typename ::std::mt19937::result_type val);                     \
  template Random< T, ::std::mt19937, I, F >::Random(int val);          \
  template Random< T, ::std::mt19937, I, F >::Random(::std::time_t val); \
  template Random< T, ::std::mt19937, I, F >::Random(::std::seed_seq q);

THUNDER_RANDOM_INSTANTIATE_MT19937(
    DoubleTensor, int, typename DoubleTensor::value_type);
THUNDER_RANDOM_INSTANTIATE_MT19937(
    FloatTensor, int, typename FloatTensor::value_type);
THUNDER_RANDOM_INSTANTIATE_MT19937(
    SizeTensor, typename SizeTensor::value_type, double);

#undef THUNDR_RANDOM_INSTANTIATE_MT19937

}  // namespace random
}  // namespace thunder
