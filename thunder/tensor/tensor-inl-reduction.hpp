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
#include <stdlib.h>

#include "thunder/exception.hpp"
#include "thunder/tensor/index_iterator.hpp"

namespace thunder {
namespace tensor {

template < typename S >
typename Tensor< S >::value_type Tensor< S >::max(
    Tensor< size_storage > *pos) const {
  value_type max_value = ::std::numeric_limits< value_type >::lowest();
  if (pos->size() != size_.size()) {
    pos->resize(size_.size());
  }
  if (partialContiguous(0, size_.size() - 1)) {
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
    for (reference_iterator begin = reference_begin(), end = reference_end();
         begin != end; ++begin) {
      if (*begin > max_value) {
          max_value = *begin;
          *pos = begin.position();
      }
    }
  }
  return max_value;
}

template < typename S >
typename Tensor< S >::value_type Tensor< S >::min(
    Tensor< size_type > *pos) const {
  value_type min_value = ::std::numeric_limits< value_type >::max();
  if (pos->size() != size_.size()) {
    pos->resize(size_.size());
  }
  if (partialContiguous(0, size_.size() - 1)) {
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
    for (reference_iterator begin = reference_begin(), end = reference_end();
         begin != end; ++begin) {
      if (*begin < min_value) {
        min_value = *begin;
        *pos = begin.position();
      }
    }
  }
  return min_value;
}

template < typename S >
typename Tensor< S >::value_type max() const {
  value_type max_value = ::std::numeric_limits< value_type >::lowest();
  if (partialContiguous(0, size_.size() - 1)) {
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
typename Tensor< S >::value_type min() const {
  value_type min_value = ::std::numeric_limits< value_type >::max();
  if (partialContiguous(0, size_.size() - 1)) {
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
  if (partialContiguous(0, size_.size() - 1)) {
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

virtual value_type prod() const {
  value_type prod_value = 1;
  if (partialContiguous(0, size_.size() - 1)) {
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
  if (partialContiguous(0, size_.size() - 1)) {
    difference_type step = stride_[stride_.size() - 1];
    pointer data_pointer = data();
    size_type data_length = length();
    for (size_type i = 0; i < data_length; ++i) {
      sum_value += data_pointer[i * step] * data_pointer[i * step];
    }
  } else {
    for (reference_iterator begin = reference_begin(), end = reference_end();
         begin != end; ++begin) {
      sum_value += (*begin) * (*begin);
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

// Reduction operations along a dimension
template < typename S >
Tensor< S > Tensor< S >::max(dim_type d, Tensor< size_storage > *pos) const {
  if (d >= size_.size()) {
    throw out_of_range("Dimension exceeds limit");
  }
  if (size_.size() == 1) {
    pos->resize(1);
    Tensor t(1);
    difference_type step = stride_[0];
    size_type data_length = size_[0];
    pointer data_pointer = data();
    size_type position = 0;
    value_type max_value = ::std::numeric_limits< value_type >::lowest();
    for (size_type i = 0; i < data_length; ++i) {
      if (data_pointer[i * step] > max_value) {
        max_value = data_pointer;
        position = i;
      }
    }
    t() = max_value;
    (*pos)() = max_value;
    return t;
  } else {
    typename Tensor< size_storage >::size_storage pos_sz(size_.size() - 1);
    for (dim_type i = 0; i < d; ++i) {
      pos_sz[i] = size_[i];
    }
    for (dim_type i = d + 1; i < size_.size(); ++i) {
      pos_sz[i - 1] = size_[i];
    }
    pos->resize(pos_sz);
    Tensor t;
    t.resizeAs(*pos);
    size_storage position;
    value_type max_value = ::std::numeric_limits< value_type >::lowest();
    if (partialContiguity(0, d) && partialContiguity(d, size_.size() - 1)) {
      // Partial contiguity
      difference_type step = stride_[d];
      difference_type left_step = d > 0 ? stride_[d-1] : 0;
      difference_type right_step = stride_[stride_.size() - 1];
      difference_type left_t_step = d > 0 ? t.stride_[d-1] : 0;
      size_type left_length = 1;
      for (dim_type i = 0; i < d; ++i) {
        left_length *= size_[i]; 
      }
      size_type right_length = 1;
      for (dim_type i = d + 1; i < size_.size(); ++i) {
        right_length *= size_[i];
      }
      size_type d_length = size_[d];
      pointer data_pointer = data();
      pointer t_data_pointer = t.data();
      pointer pos_data_pointer = pos->data();
      for (size_type left = 0; left < left_length; ++left) {
        for (size_type right = 0; right < right_length; ++right) {
          max_value = ::std::numeric_limits< value_type >::lowest();
          for (size_type i = 0; i < d_length; ++i) {
            if (data_pointer[left * left_step + right * right_step + i * step] >
                max_value) {
              max_value = data_pointer[left * left_step +
                                       right * right_step + i * step];
              pos_data_pointer[left * left_t_step + right] = i;
            }
          }
          t_data_pointer[left * left_t_step + right] = i;
        }
      }
    } else {
      // No partial contiguity
      for (IndexIterator< size_storage > begin =
               IndexIterator< size_storage >::begin(pos_sz),
               end = IndexIterator < size_storage >::end(pos_sz);
           begin != end; ++begin) {
        for (dim_type i = 0; i < d; ++i) {
          position[i] = (*begin)[i];
        }
        for (dim_type i = d + 1; i < size_.size(); ++i) {
          position[i] = (*begin)[i - 1];
        }
        max_value = ::std::numeric_limits< value_type >::lowest();
        for (size_type i = 0; i < size_[d]; ++i) {
          position[d] = i;
          if ((*this)(position) > max_value) {
            max_value = (*this)(position);
            (*pos)(*begin) = i;
          }
        }
        t(*begin) = max_value;
      }
    }
    return t;
  }
}

template < typename S >
Tensor< S > Tensor< S >::min(dim_type d, Tensor< size_storage > *pos) const {
}

template < typename S >
Tensor< S > Tensor< S >::max(dim_type d) const {
}

virtual Tensor min(dim_type d) const;
virtual Tensor sum(dim_type d) const;
virtual Tensor prod(dim_type d) const;
virtual Tensor mean(dim_type d) const;
virtual Tensor var(dim_type d) const;
virtual Tensor std(dim_type d) const;

// Static reduction operations are deligated
static Tensor max(const Tensor &x, dim_type d, size_storage *pos);
static Tensor min(const Tensor &x, dim_type d, size_storage *pos);
static Tensor max(const Tensor &x, dim_type d);
static Tensor min(const Tensor &x, dim_type d);
static Tensor sum(const Tensor &x, dim_type d);
static Tensor prod(const Tensor &x, dim_type d);
static Tensor mean(const Tensor &x, dim_type d);
static Tensor var(const Tensor &x, dim_type d);
static Tensor std(const Tensor &x, dim_type d);

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_REDUCTION_HPP_
