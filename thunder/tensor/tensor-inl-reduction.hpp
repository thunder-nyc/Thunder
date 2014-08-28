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

namespace thunder {
namespace tensor {

template < typename S >
typename Tensor< S >::value_type max(size_storage *pos = nullptr) const {
  value_type max_value = ::std::numeric_limits< value_type >::lowest()
  if (pos == nullptr) {
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
  } else {
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
        (*pos)[i] = position % size_[i];
        position /= size_[i];
      }
      (*pos)[0] = position;
    } else {
      for (reference_iterator begin = reference_begin(), end = reference_end();
           begin != end; ++begin) {
        if (*begin > max_value) {
          max_value = *begin;
          *pos = begin.position();
        }
      }
    }
  }
  return max_value;
}

template < typename S >
typename Tensor< S >::value_type min(size_storage *pos = nullptr) const {
  value_type min_value = ::std::numeric_limits< value_type >::max()
  if (pos == nullptr) {
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
  } else {
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
        (*pos)[i] = position % size_[i];
        position /= size_[i];
      }
      (*pos)[0] = position;
    } else {
      for (reference_iterator begin = reference_begin(), end = reference_end();
           begin != end; ++begin) {
        if (*begin < min_value) {
          min_value = *begin;
          *pos = begin.position();
        }
      }
    }
  }
  return min_value;
}

virtual value_type sum() const;
virtual value_type prod() const;
virtual value_type mean() const;
virtual value_type var() const;
virtual value_type std() const;

static value_type max(const Tensor &x, size_storage *pos = nullptr);
static value_type min(const Tensor &x, size_storage *pos = nullptr);
static value_type sum(const Tensor &x);
static value_type prod(const Tensor &x);
static value_type mean(const Tensor &x);
static value_type var(const Tensor &x);
static value_type std(const Tensor &x);

virtual value_type max(dim_type d, size_storage *pos = nullptr) const;
virtual value_type min(dim_type d, size_storage *pos = nullptr) const;
virtual value_type sum(dim_type d) const;
virtual value_type prod(dim_type d) const;
virtual value_type mean(dim_type d) const;
virtual value_type var(dim_type d) const;
virtual value_type std(dim_type d) const;

static value_type max(const Tensor &x, dim_type d,
                      size_storage *pos = nullptr);
static value_type min(const Tensor &x, dim_type d,
                      size_storage *pos = nullptr);
static value_type sum(const Tensor &x, dim_type d);
static value_type prod(const Tensor &x, dim_type d);
static value_type mean(const Tensor &x, dim_type d);
static value_type var(const Tensor &x, dim_type d);
static value_type std(const Tensor &x, dim_type d);

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_REDUCTION_HPP_
