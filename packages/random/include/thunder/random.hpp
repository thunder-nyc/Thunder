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

#ifndef THUNDER_RANDOM_HPP_
#define THUNDER_RANDOM_HPP_

#include "thunder/random/random.hpp"

#include <random>

#include "thunder/random/math.hpp"
#include "thunder/tensor.hpp"

namespace thunder {

template < typename T = DoubleTensor, typename G = ::std::mt19937,
           typename I = int, typename F = double >
using Random = random::Random< T, G, I, F >;

typedef Random<
  DoubleTensor, ::std::mt19937, int, typename DoubleTensor::value_type >
DoubleRandom;
typedef Random<
  FloatTensor, ::std::mt19937, int, typename FloatTensor::value_type >
FloatRandom;
typedef Random<
  SizeTensor, ::std::mt19937, typename SizeTensor::value_type, double >
SizeRandom;

}  // namespace thunder

namespace thunder {
namespace random {

#define THUNDER_RANDOM_INSTANTIATE_MT19937(T, I, F)                     \
  extern template class Random< T, ::std::mt19937, I, F >;              \
  extern template Random< T, ::std::mt19937, I, F >::Random();          \
  extern template Random< T, ::std::mt19937, I, F >::Random::           \
  Random(typename ::std::mt19937::result_type val);                     \
  extern template Random< T, ::std::mt19937, I, F >::Random::           \
  Random(::std::seed_seq q);

THUNDER_RANDOM_INSTANTIATE_MT19937(
    DoubleTensor, int, typename DoubleTensor::value_type);
THUNDER_RANDOM_INSTANTIATE_MT19937(
    FloatTensor, int, typename FloatTensor::value_type);
THUNDER_RANDOM_INSTANTIATE_MT19937(
    SizeTensor, typename SizeTensor::value_type, double);

#undef THUNDR_RANDOM_INSTANTIATE_MT19937

}  // namespace random
}  // namespace thunder

#endif  // THUNDER_RANDOM_HPP_
