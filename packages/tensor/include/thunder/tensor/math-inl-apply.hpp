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

#ifndef THUNDER_TENSOR_MATH_INL_APPLY_HPP_
#define THUNDER_TENSOR_MATH_INL_APPLY_HPP_

#include "thunder/tensor/math.hpp"
#include "thunder/tensor/math-inl.hpp"

#include <functional>

#include "thunder/exception.hpp"
#include "thunder/tensor/index_iterator.hpp"

namespace thunder {
namespace tensor {
namespace math {

template < typename T >
const T& apply(const T &x, const ::std::function< typename T::value_type(
    typename T::value_type) > &lambda) {
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::pointer data_pointer = x.data();
    typename T::size_type data_length = x.length();
    typename T::difference_type step = x.stride(x.dimension() - 1);
    for (typename T::size_type i = 0; i < data_length; ++i) {
      data_pointer[i * step] = lambda(data_pointer[i * step]);
    }
  } else {
    for (typename T::reference_iterator begin = x.reference_begin(),
             end = x.reference_end(); begin != end; ++begin) {
      *begin = lambda(*begin);
    }
  }
  return x;
}

template < typename T >
const T& apply(const T &x, const ::std::function< typename T::value_type(
    const typename T::value_type&) > &lambda) {
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::pointer data_pointer = x.data();
    typename T::size_type data_length = x.length();
    typename T::difference_type step = x.stride(x.dimension() - 1);
    for (typename T::size_type i = 0; i < data_length; ++i) {
      data_pointer[i * step] = lambda(data_pointer[i * step]);
    }
  } else {
    for (typename T::reference_iterator begin = x.reference_begin(),
             end = x.reference_end(); begin != end; ++begin) {
      *begin = lambda(*begin);
    }
  }
  return x;
}

template < typename T >
const T& apply(const T &x, const ::std::function< void(
    typename T::value_type&) > &lambda) {
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::pointer data_pointer = x.data();
    typename T::size_type data_length = x.length();
    typename T::difference_type step = x.stride(x.dimension() - 1);
    for (typename T::size_type i = 0; i < data_length; ++i) {
      lambda(data_pointer[i * step]);
    }
  } else {
    for (typename T::reference_iterator begin = x.reference_begin(),
             end = x.reference_end(); begin != end; ++begin) {
      lambda(*begin);
    }
  }
  return x;
}

template < typename T >
const T& apply(const T &x, const ::std::function< void(
    typename T::value_type*) > &lambda) {
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::pointer data_pointer = x.data();
    typename T::size_type data_length = x.length();
    typename T::difference_type step = x.stride(x.dimension() - 1);
    for (typename T::size_type i = 0; i < data_length; ++i) {
      lambda(&data_pointer[i * step]);
    }
  } else {
    for (typename T::reference_iterator begin = x.reference_begin(),
             end = x.reference_end(); begin != end; ++begin) {
      lambda(&(*begin));
    }
  }
  return x;
}

}  // namespace math
}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_MATH_INL_APPLY_HPP_
