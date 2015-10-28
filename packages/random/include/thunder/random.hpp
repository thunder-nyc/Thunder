/*
 * \copyright Copyright 2014-2015 Xiang Zhang All Rights Reserved.
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

#include <ctime>
#include <random>

#include "thunder/random/tensor_type.hpp"
#include "thunder/tensor.hpp"

namespace thunder {

template < typename T,
           typename G = typename random::TensorType< T >::generator_type,
           typename I = typename random::TensorType< T >::integer_type,
           typename F = typename random::TensorType< T >::float_type >
using Random = random::Random< T, G, I, F >;

typedef Random< DoubleTensor > DoubleRandom;
typedef Random< FloatTensor > FloatRandom;
typedef Random< SizeTensor > SizeRandom;

}  // namespace thunder

namespace thunder {
namespace random {

#define THUNDER_RANDOM_INSTANTIATE(T)                             \
  extern template class Random< T >;                              \
  extern template Random< T >::Random();                          \
  extern template Random< T >::Random(                            \
      typename ::std::mt19937::result_type val);                  \
  extern template Random< T >::Random(int val);                   \
  extern template Random< T >::Random(::std::time_t val);       \
  extern template Random< T >::Random(::std::seed_seq q);

THUNDER_RANDOM_INSTANTIATE(DoubleTensor);
THUNDER_RANDOM_INSTANTIATE(FloatTensor);
THUNDER_RANDOM_INSTANTIATE(SizeTensor);

#undef THUNDR_RANDOM_INSTANTIATE

}  // namespace random
}  // namespace thunder

#endif  // THUNDER_RANDOM_HPP_
