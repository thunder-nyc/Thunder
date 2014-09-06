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

#ifndef THUNDER_TENSOR_TENSOR_INL_REDUCTION_HPP_
#define THUNDER_TENSOR_TENSOR_INL_REDUCTION_HPP_

#include "thunder/tensor/tensor.hpp"
#include "thunder/tensor/tensor-inl.hpp"

#include <limits>
#include <cmath>

#include "thunder/exception.hpp"
#include "thunder/tensor/index_iterator.hpp"

namespace thunder {
namespace tensor {

template < typename S >
typename Tensor< S >::value_type Tensor< S >::max(
    Tensor< size_storage > *pos) const {
  value_type max_value = ::std::numeric_limits< value_type >::lowest();
  if (pos->size(0) != size_.size()) {
    pos->resize(size_.size());
  }
  if (partialContiguity(0, size_.size() - 1)) {
    difference_type step = stride_[stride_.size() - 1];
    pointer data_pointer = data();
    size_type data_length = length();
    size_type position = 0;
    for (size_type i = 0; i < data_length; ++i) {
      if (data_pointer[i * step] > max_value) {
        max_value = data_pointer[i * step];
        position = i;
      }
    }
    for (dim_type i = size_.size() - 1; i > 0; --i) {
      (*pos)(i) = position % size_[i];
      position /= size_[i];
    }
    (*pos)(0) = position;
  } else {
    size_storage position;
    for (reference_iterator begin = reference_begin(), end = reference_end();
         begin != end; ++begin) {
      if (*begin > max_value) {
          max_value = *begin;
          position = begin.position();
      }
    }
    for (dim_type i = 0; i < size_.size(); ++i) {
      (*pos)(i) = position[i];
    }
  }
  return max_value;
}

template < typename S >
typename Tensor< S >::value_type Tensor< S >::min(
    Tensor< size_storage > *pos) const {
  value_type min_value = ::std::numeric_limits< value_type >::max();
  if (pos->size(0) != size_.size()) {
    pos->resize(size_.size());
  }
  if (partialContiguity(0, size_.size() - 1)) {
    difference_type step = stride_[stride_.size() - 1];
    pointer data_pointer = data();
    size_type data_length = length();
    size_type position = 0;
    for (size_type i = 0; i < data_length; ++i) {
      if (data_pointer[i * step] < min_value) {
        min_value = data_pointer[i * step];
        position = i;
      }
    }
    for (dim_type i = size_.size() - 1; i > 0; --i) {
      (*pos)(i) = position % size_[i];
      position /= size_[i];
    }
    (*pos)(0) = position;
  } else {
    size_storage position;
    for (reference_iterator begin = reference_begin(), end = reference_end();
         begin != end; ++begin) {
      if (*begin < min_value) {
        min_value = *begin;
        position = begin.position();
      }
    }
    for (dim_type i = 0; i < size_.size(); ++i) {
      (*pos)(i) = position[i];
    }
  }
  return min_value;
}

template < typename S >
typename Tensor< S >::value_type Tensor< S >::max() const {
  value_type max_value = ::std::numeric_limits< value_type >::lowest();
  if (partialContiguity(0, size_.size() - 1)) {
    difference_type step = stride_[stride_.size() - 1];
    pointer data_pointer = data();
    size_type data_length = length();
    for (size_type i = 0; i < data_length; ++i) {
      if (data_pointer[i * step] > max_value) {
        max_value = data_pointer[i * step];
      }
    }
  } else {
    for (reference_iterator begin = reference_begin(), end = reference_end();
         begin != end; ++begin) {
      if (*begin > max_value) {
        max_value = *begin;
      }
    }
  }
  return max_value;
}

template < typename S >
typename Tensor< S >::value_type Tensor< S >::min() const {
  value_type min_value = ::std::numeric_limits< value_type >::max();
  if (partialContiguity(0, size_.size() - 1)) {
    difference_type step = stride_[stride_.size() - 1];
    pointer data_pointer = data();
    size_type data_length = length();
    for (size_type i = 0; i < data_length; ++i) {
      if (data_pointer[i * step] < min_value) {
        min_value = data_pointer[i * step];
      }
    }
  } else {
    for (reference_iterator begin = reference_begin(), end = reference_end();
         begin != end; ++begin) {
      if (*begin < min_value) {
        min_value = *begin;
      }
    }
  }
  return min_value;
}

template < typename S >
typename Tensor< S >::value_type Tensor< S >::sum() const {
  value_type sum_value = 0;
  if (partialContiguity(0, size_.size() - 1)) {
    difference_type step = stride_[stride_.size() - 1];
    pointer data_pointer = data();
    size_type data_length = length();
    for (size_type i = 0; i < data_length; ++i) {
      sum_value += data_pointer[i * step];
    }
  } else {
    for (reference_iterator begin = reference_begin(), end = reference_end();
         begin != end; ++begin) {
      sum_value += *begin;
    }
  }
  return sum_value;
}

template < typename S >
typename Tensor< S >::value_type Tensor< S >::prod() const {
  value_type prod_value = 1;
  if (partialContiguity(0, size_.size() - 1)) {
    difference_type step = stride_[stride_.size() - 1];
    pointer data_pointer = data();
    size_type data_length = length();
    for (size_type i = 0; i < data_length; ++i) {
      prod_value *= data_pointer[i * step];
    }
  } else {
    for (reference_iterator begin = reference_begin(), end = reference_end();
         begin != end; ++begin) {
      prod_value *= *begin;
    }
  }
  return prod_value;
}

template < typename S >
typename Tensor< S >::value_type Tensor< S >::mean() const {
  return sum() / static_cast< value_type >(length());
}

template < typename S >
typename Tensor< S >::value_type Tensor< S >::var() const {
  value_type sum_value = 0;
  value_type mean_value = mean();
  size_type data_length = length();
  if (partialContiguity(0, size_.size() - 1)) {
    difference_type step = stride_[stride_.size() - 1];
    pointer data_pointer = data();
    for (size_type i = 0; i < data_length; ++i) {
      sum_value += (data_pointer[i * step] - mean_value) *
          (data_pointer[i * step] - mean_value);
    }
  } else {
    for (reference_iterator begin = reference_begin(), end = reference_end();
         begin != end; ++begin) {
      sum_value += (*begin - mean_value) * (*begin - mean_value);
    }
  }
  return sum_value / static_cast< value_type >(data_length);
}

template < typename S >
typename Tensor< S >::value_type Tensor< S >::max(
    const Tensor &x, Tensor< size_storage > *pos) {
  return x.max(pos);
}

template < typename S >
typename Tensor< S >::value_type Tensor< S >::min(
    const Tensor &x, Tensor< size_storage > *pos) {
  return x.min(pos);
}

template < typename S >
typename Tensor< S >::value_type Tensor< S >::std() const {
  return ::std::sqrt(var());
}

template < typename S >
typename Tensor< S >::value_type Tensor< S >::max(const Tensor &x) {
  return x.max();
}

template < typename S >
typename Tensor< S >::value_type Tensor< S >::min(const Tensor &x) {
  return x.min();
}

template < typename S >
typename Tensor< S >::value_type Tensor< S >::sum(const Tensor &x) {
  return x.sum();
}

template < typename S >
typename Tensor< S >::value_type Tensor< S >::prod(const Tensor &x) {
  return x.prod();
}

template < typename S >
typename Tensor< S >::value_type Tensor< S >::mean(const Tensor &x) {
  return x.mean();
}

template < typename S >
typename Tensor< S >::value_type Tensor< S >::var(const Tensor &x) {
  return x.var();
}

template < typename S >
typename Tensor< S >::value_type Tensor< S >::std(const Tensor &x) {
  return x.std();
}

template < typename S >
Tensor< S > Tensor< S >::max(dim_type d, Tensor< size_storage > *pos) const {
  return Tensor();
}

template < typename S >
Tensor< S > Tensor< S >::min(dim_type d, Tensor< size_storage > *pos) const {
  return Tensor();
}

template < typename S >
Tensor< S > Tensor< S >::max(dim_type d) const {
  if (d >= size_.size()) {
    throw out_of_range("Dimension exceeds limit");
  }
  size_storage sz = size_;
  sz[d] = 1;
  Tensor t(sz);
  if (partialContiguity(0, d) && partialContiguity(d + 1, size_.size() - 1)) {
    // Get data pointers
    pointer x_data = data();
    pointer t_data = t.data();
    // Determine left and right size
    size_type x_left_length = 1;
    for (dim_type i = 0; i < d; ++i) {
      x_left_length *= size_[i];
    }
    size_type x_length = size_[d];
    size_type x_right_length = 1;
    for (dim_type i = d + 1; i < size_.size(); ++i) {
      x_right_length *= size_[i];
    }
    // Determine left and right step
    difference_type x_left_step = d > 0 ? stride_[d - 1] : 0;
    difference_type x_step = stride_[d];
    difference_type x_right_step = stride_[stride_.size() - 1];
    difference_type t_left_step = d > 0 ? t.stride(d - 1) : 0;
    difference_type t_right_step = t.stride(t.dimension() - 1);
    // Run the loop
    value_type max_value = ::std::numeric_limits< value_type >::lowest();
    for (size_type i = 0; i < x_left_length; ++i) {
      for (size_type j = 0; j < x_right_length; ++j) {
        max_value = ::std::numeric_limits< value_type >::lowest();
        for (size_type k = 0; k < x_length; ++k) {
          value_type current_value =
              x_data[i * x_left_step + j * x_right_step + k * x_step];
          if (current_value > max_value) {
            max_value = current_value;
          }
        }
        t_data[i * t_left_step + j * t_right_step] = max_value;
      }
    }
  } else {
    size_storage position;
    size_type size_d = size_[d];
    value_type max_value = ::std::numeric_limits< value_type >::lowest();
    for (reference_iterator begin = t.reference_begin(),
             end = t.reference_end(); begin != end; ++begin) {
      max_value = ::std::numeric_limits< value_type >::lowest();
      position = begin.position();
      for (size_type i = 0; i < size_d; ++i) {
        position[d] = i;
        value_type current_value = (*this)(position);
        if (current_value > max_value) {
          max_value = current_value;
        }
      }
      *begin = max_value;
    }
  }
  return t;
}

template < typename S >
Tensor< S > Tensor< S >::min(dim_type d) const {
  return Tensor();
}

template < typename S >
Tensor< S > Tensor< S >::sum(dim_type d) const {
  return Tensor();
}

template < typename S >
Tensor< S > Tensor< S >::prod(dim_type d) const {
  return Tensor();
}

template < typename S >
Tensor< S > Tensor< S >::mean(dim_type d) const {
  return Tensor();
}

template < typename S >
Tensor< S > Tensor< S >::var(dim_type d) const {
  return Tensor();
}

template < typename S >
Tensor< S > Tensor< S >::std(dim_type d) const {
  return Tensor();
}

template < typename S >
Tensor< S > Tensor< S >::max(
    const Tensor &x, dim_type d, Tensor< size_storage > *pos) {
  return x.max(d, pos);
}

template < typename S >
Tensor< S > Tensor< S >::min(
    const Tensor &x, dim_type d, Tensor< size_storage > *pos) {
  return x.min(d, pos);
}

template < typename S >
Tensor< S > Tensor< S >::max(const Tensor &x, dim_type d) {
  return x.max(d);
}

template < typename S >
Tensor< S > Tensor< S >::min(const Tensor &x, dim_type d) {
  return x.min(d);
}

template < typename S >
Tensor< S > Tensor< S >::sum(const Tensor &x, dim_type d) {
  return x.sum(d);
}

template < typename S >
Tensor< S > Tensor< S >::prod(const Tensor &x, dim_type d) {
  return x.prod(d);
}

template < typename S >
Tensor< S > Tensor< S >::mean(const Tensor &x, dim_type d) {
  return x.mean(d);
}

template < typename S >
Tensor< S > Tensor< S >::var(const Tensor &x, dim_type d) {
  return x.var(d);
}

template < typename S >
Tensor< S > Tensor< S >::std(const Tensor &x, dim_type d) {
  return x.std(d);
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_REDUCTION_HPP_
