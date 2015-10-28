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

#ifndef THUNDER_RANDOM_TENSOR_TYPE_HPP_
#define THUNDER_RANDOM_TENSOR_TYPE_HPP_

#include <random>

#include "thunder/tensor.hpp"

namespace thunder {
namespace random {

template < typename T >
class TensorType {
 public:
  typedef T tensor_type;
  typedef ::std::mt19937 generator_type;
  typedef int integer_type;
  typedef double float_type;
};

template < >
class TensorType< DoubleTensor > {
 public:
  typedef DoubleTensor tensor_type;
  typedef ::std::mt19937 generator_type;
  typedef int integer_type;
  typedef typename DoubleTensor::value_type float_type;
};

template < >
class TensorType< FloatTensor > {
 public:
  typedef FloatTensor tensor_type;
  typedef ::std::mt19937 generator_type;
  typedef int integer_type;
  typedef typename FloatTensor::value_type float_type;
};

template < >
class TensorType< SizeTensor > {
 public:
  typedef SizeTensor tensor_type;
  typedef ::std::mt19937 generator_type;
  typedef typename SizeTensor::value_type integer_type;
  typedef double float_type;
};

}  // namespace random
}  // namespace thunder

#endif  // THUNDER_RANDOM_TENSOR_TYPE_HPP_
