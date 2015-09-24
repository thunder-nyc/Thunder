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

#ifndef THUNDER_TENSOR_MATH_INL_REDUCTION_HPP_
#define THUNDER_TENSOR_MATH_INL_REDUCTION_HPP_

#include "thunder/tensor/math.hpp"
#include "thunder/tensor/math-inl.hpp"

#include <cmath>
#include <limits>

#include "thunder/exception.hpp"
#include "thunder/tensor/index_iterator.hpp"
#include "thunder/tensor/tensor.hpp"

namespace thunder {
namespace tensor {
namespace math {

template < typename T >
const typename T::value_type max(
    const T &x, Tensor< typename T::size_storage > *pos) {
  typename T::value_type max_value =
      ::std::numeric_limits< typename T::value_type >::lowest();
  if (pos->size(0) != x.dimension()) {
    pos->resize(x.dimension());
  }
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::pointer x_pointer = x.data();
    typename T::size_type x_length = x.length();
    typename T::size_type position = 0;
    for (typename T::size_type i = 0; i < x_length; ++i) {
      if (x_pointer[i * x_step] > max_value) {
        max_value = x_pointer[i * x_step];
        position = i;
      }
    }
    for (typename T::dim_type i = x.dimension() - 1; i > 0; --i) {
      (*pos)(i) = position % x.size(i);
      position /= x.size(i);
    }
    (*pos)(0) = position;
  } else {
    typename T::size_storage position;
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      if (*x_begin > max_value) {
        max_value = *x_begin;
        position = x_begin.position();
      }
    }
    for (typename T::dim_type i = 0; i < x.dimension(); ++i) {
      (*pos)(i) = position[i];
    }
  }
  return max_value;
}

template < typename T >
const typename T::value_type min(
    const T &x, Tensor< typename T::size_storage > *pos) {
  typename T::value_type min_value =
      ::std::numeric_limits< typename T::value_type >::max();
  if (pos->size(0) != x.dimension()) {
    pos->resize(x.dimension());
  }
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::pointer x_pointer = x.data();
    typename T::size_type x_length = x.length();
    typename T::size_type position = 0;
    for (typename T::size_type i = 0; i < x_length; ++i) {
      if (x_pointer[i * x_step] < min_value) {
        min_value = x_pointer[i * x_step];
        position = i;
      }
    }
    for (typename T::dim_type i = x.dimension() - 1; i > 0; --i) {
      (*pos)(i) = position % x.size(i);
      position /= x.size(i);
    }
    (*pos)(0) = position;
  } else {
    typename T::size_storage position;
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      if (*x_begin < min_value) {
        min_value = *x_begin;
        position = x_begin.position();
      }
    }
    for (typename T::dim_type i = 0; i < x.dimension(); ++i) {
      (*pos)(i) = position[i];
    }
  }
  return min_value;
}

template < typename T >
const typename T::value_type max(const T &x) {
  typename T::value_type max_value =
      ::std::numeric_limits< typename T::value_type >::lowest();
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::pointer x_pointer = x.data();
    typename T::size_type x_length = x.length();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      if (x_pointer[i * x_step] > max_value) {
        max_value = x_pointer[i * x_step];
      }
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      if (*x_begin > max_value) {
        max_value = *x_begin;
      }
    }
  }
  return max_value;
}

template < typename T >
const typename T::value_type min(const T &x) {
  typename T::value_type min_value =
      ::std::numeric_limits< typename T::value_type >::max();
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::pointer x_pointer = x.data();
    typename T::size_type x_length = x.length();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      if (x_pointer[i * x_step] < min_value) {
        min_value = x_pointer[i * x_step];
      }
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      if (*x_begin < min_value) {
        min_value = *x_begin;
      }
    }
  }
  return min_value;
}

template < typename T >
const typename T::value_type sum(const T &x) {
  typename T::value_type sum_value = 0;
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::pointer x_pointer = x.data();
    typename T::size_type x_length = x.length();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      sum_value += x_pointer[i * x_step];
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      sum_value += *x_begin;
    }
  }
  return sum_value;
}

template < typename T >
const typename T::value_type prod(const T &x) {
  typename T::value_type prod_value = 1;
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::pointer x_pointer = x.data();
    typename T::size_type x_length = x.length();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      prod_value *= x_pointer[i * x_step];
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      prod_value *= *x_begin;
    }
  }
  return prod_value;
}

template < typename T >
const typename T::value_type mean(const T &x) {
  return x.sum() / static_cast< typename T::value_type >(x.length());
}

template < typename T >
const typename T::value_type var(const T &x) {
  typename T::value_type sum_value = 0;
  typename T::value_type mean_value = x.mean();
  typename T::size_type x_length = x.length();
  if (x.partialContiguity(0, x.dimension() - 1)) {
    typename T::difference_type x_step = x.stride(x.dimension() - 1);
    typename T::pointer x_pointer = x.data();
    for (typename T::size_type i = 0; i < x_length; ++i) {
      sum_value += (x_pointer[i * x_step] - mean_value) *
          (x_pointer[i * x_step] - mean_value);
    }
  } else {
    for (typename T::reference_iterator x_begin = x.reference_begin(),
             x_end = x.reference_end(); x_begin != x_end; ++x_begin) {
      sum_value += (*x_begin - mean_value) * (*x_begin - mean_value);
    }
  }
  return sum_value / static_cast< typename T::value_type >(x_length);
}

template < typename T >
const typename T::value_type std(const T &x) {
  return ::std::sqrt(x.var());
}

template < typename T >
T max(const T &x, typename T::dim_type d,
      Tensor< typename T::size_storage > *pos) {
  if (d >= x.dimension()) {
    throw out_of_range("Dimension exceeds limit.");
  }
  typename T::size_storage sz = x.size();
  sz[d] = 1;
  T t(sz);
  pos->resizeAs(t);
  if (x.partialContiguity(0, d > 0 ? (d - 1) : 0)
      && x.partialContiguity(d + 1, x.dimension() - 1)
      && pos->partialContiguity(0, d > 0 ? (d - 1) : 0)
      && pos->partialContiguity(d + 1, x.dimension() - 1)) {
    // Get data pointers
    typename T::pointer x_data = x.data();
    typename T::pointer t_data = t.data();
    typename Tensor< typename T::size_storage >::pointer p_data = pos->data();
    // Determine left and right size
    typename T::size_type x_left_length = 1;
    for (typename T::dim_type i = 0; i < d; ++i) {
      x_left_length *= x.size(i);
    }
    typename T::size_type x_length = x.size(d);
    typename T::size_type x_right_length = 1;
    for (typename T::dim_type i = d + 1; i < x.dimension(); ++i) {
      x_right_length *= x.size(i);
    }
    // Determine left and right step
    typename T::difference_type x_left_step = d > 0 ? x.stride(d - 1) : 0;
    typename T::difference_type x_step = x.stride(d);
    typename T::difference_type x_right_step = x.stride(x.dimension() - 1);
    typename T::difference_type t_left_step = d > 0 ? t.stride(d - 1) : 0;
    typename T::difference_type t_right_step = t.stride(t.dimension() - 1);
    typename T::difference_type p_left_step = d > 0 ? pos->stride(d - 1) : 0;
    typename T::difference_type p_right_step =
        pos->stride(pos->dimension() - 1);
    // Run the loop
    typename T::value_type max_value =
        ::std::numeric_limits< typename T::value_type >::lowest();
    typename T::size_type pos_value = 0;
    for (typename T::size_type i = 0; i < x_left_length; ++i) {
      for (typename T::size_type j = 0; j < x_right_length; ++j) {
        max_value = ::std::numeric_limits< typename T::value_type >::lowest();
        pos_value = 0;
        for (typename T::size_type k = 0; k < x_length; ++k) {
          typename T::value_type current_value =
              x_data[i * x_left_step + j * x_right_step + k * x_step];
          if (current_value > max_value) {
            max_value = current_value;
            pos_value = k;
          }
        }
        t_data[i * t_left_step + j * t_right_step] = max_value;
        p_data[i * p_left_step + j * p_right_step] = pos_value;
      }
    }
  } else {
    typename T::size_storage position;
    typename T::size_type size_d = x.size(d);
    typename T::value_type max_value =
        ::std::numeric_limits< typename T::value_type >::lowest();
    typename T::size_type pos_value = 0;
    typename Tensor< typename T::size_storage >::reference_iterator p_begin =
        pos->reference_begin();
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end();
         t_begin != t_end; ++t_begin, ++p_begin) {
      max_value = ::std::numeric_limits< typename T::value_type >::lowest();
      pos_value = 0;
      position = t_begin.position();
      for (typename T::size_type i = 0; i < size_d; ++i) {
        position[d] = i;
        typename T::value_type current_value = x(position);
        if (current_value > max_value) {
          max_value = current_value;
          pos_value = i;
        }
      }
      *t_begin = max_value;
      *p_begin = pos_value;
    }
  }
  return t;
}

template < typename T >
T min(const T &x, typename T::dim_type d,
      Tensor< typename T::size_storage > *pos) {
  if (d >= x.dimension()) {
    throw out_of_range("Dimension exceeds limit.");
  }
  typename T::size_storage sz = x.size();
  sz[d] = 1;
  T t(sz);
  pos->resizeAs(t);
  if (x.partialContiguity(0, d > 0 ? (d - 1) : 0)
      && x.partialContiguity(d + 1, x.dimension() - 1)
      && pos->partialContiguity(0, d > 0 ? (d - 1) : 0)
      && pos->partialContiguity(d + 1, x.dimension() - 1)) {
    // Get data pointers
    typename T::pointer x_data = x.data();
    typename T::pointer t_data = t.data();
    typename Tensor< typename T::size_storage >::pointer p_data = pos->data();
    // Determine left and right size
    typename T::size_type x_left_length = 1;
    for (typename T::dim_type i = 0; i < d; ++i) {
      x_left_length *= x.size(i);
    }
    typename T::size_type x_length = x.size(d);
    typename T::size_type x_right_length = 1;
    for (typename T::dim_type i = d + 1; i < x.dimension(); ++i) {
      x_right_length *= x.size(i);
    }
    // Determine left and right step
    typename T::difference_type x_left_step = d > 0 ? x.stride(d - 1) : 0;
    typename T::difference_type x_step = x.stride(d);
    typename T::difference_type x_right_step = x.stride(x.dimension() - 1);
    typename T::difference_type t_left_step = d > 0 ? t.stride(d - 1) : 0;
    typename T::difference_type t_right_step = t.stride(t.dimension() - 1);
    typename T::difference_type p_left_step = d > 0 ? pos->stride(d - 1) : 0;
    typename T::difference_type p_right_step =
        pos->stride(pos->dimension() - 1);
    // Run the loop
    typename T::value_type min_value =
        ::std::numeric_limits< typename T::value_type >::max();
    typename T::size_type pos_value = 0;
    typename T::value_type current_value = 0;
    for (typename T::size_type i = 0; i < x_left_length; ++i) {
      for (typename T::size_type j = 0; j < x_right_length; ++j) {
        min_value = ::std::numeric_limits< typename T::value_type >::max();
        pos_value = 0;
        for (typename T::size_type k = 0; k < x_length; ++k) {
          current_value =
              x_data[i * x_left_step + j * x_right_step + k * x_step];
          if (current_value < min_value) {
            min_value = current_value;
            pos_value = k;
          }
        }
        t_data[i * t_left_step + j * t_right_step] = min_value;
        p_data[i * p_left_step + j * p_right_step] = pos_value;
      }
    }
  } else {
    typename T::size_storage position;
    typename T::size_type size_d = x.size(d);
    typename T::value_type min_value =
        ::std::numeric_limits< typename T::value_type >::max();
    typename T::size_type pos_value = 0;
    typename Tensor< typename T::size_storage >::reference_iterator p_begin =
        pos->reference_begin();
    typename T::value_type current_value = 0;
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end();
         t_begin != t_end; ++t_begin, ++p_begin) {
      min_value = ::std::numeric_limits< typename T::value_type >::max();
      pos_value = 0;
      position = t_begin.position();
      for (typename T::size_type i = 0; i < size_d; ++i) {
        position[d] = i;
        current_value = x(position);
        if (current_value < min_value) {
          min_value = current_value;
          pos_value = i;
        }
      }
      *t_begin = min_value;
      *p_begin = pos_value;
    }
  }
  return t;
}

template < typename T >
T max(const T &x, typename T::dim_type d) {
  if (d >= x.dimension()) {
    throw out_of_range("Dimension exceeds limit.");
  }
  typename T::size_storage sz = x.size();
  sz[d] = 1;
  T t(sz);
  if (x.partialContiguity(0, d > 0 ? (d - 1) : 0)
      && x.partialContiguity(d + 1, x.dimension() - 1)) {
    // Get data pointers
    typename T::pointer x_data = x.data();
    typename T::pointer t_data = t.data();
    // Determine left and right size
    typename T::size_type x_left_length = 1;
    for (typename T::dim_type i = 0; i < d; ++i) {
      x_left_length *= x.size(i);
    }
    typename T::size_type x_length = x.size(d);
    typename T::size_type x_right_length = 1;
    for (typename T::dim_type i = d + 1; i < x.dimension(); ++i) {
      x_right_length *= x.size(i);
    }
    // Determine left and right step
    typename T::difference_type x_left_step = d > 0 ? x.stride(d - 1) : 0;
    typename T::difference_type x_step = x.stride(d);
    typename T::difference_type x_right_step = x.stride(x.dimension() - 1);
    typename T::difference_type t_left_step = d > 0 ? t.stride(d - 1) : 0;
    typename T::difference_type t_right_step = t.stride(t.dimension() - 1);
    typename T::value_type max_value =
        ::std::numeric_limits< typename T::value_type >::lowest();
    typename T::value_type current_value = 0;
    for (typename T::size_type i = 0; i < x_left_length; ++i) {
      for (typename T::size_type j = 0; j < x_right_length; ++j) {
        max_value = ::std::numeric_limits< typename T::value_type >::lowest();
        for (typename T::size_type k = 0; k < x_length; ++k) {
          current_value =
              x_data[i * x_left_step + j * x_right_step + k * x_step];
          if (current_value > max_value) {
            max_value = current_value;
          }
        }
        t_data[i * t_left_step + j * t_right_step] = max_value;
      }
    }
  } else {
    typename T::size_storage position;
    typename T::size_type size_d = x.size(d);
    typename T::value_type max_value =
        ::std::numeric_limits< typename T::value_type >::lowest();
    typename T::value_type current_value = 0;
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      max_value = ::std::numeric_limits< typename T::value_type >::lowest();
      position = t_begin.position();
      for (typename T::size_type i = 0; i < size_d; ++i) {
        position[d] = i;
        current_value = x(position);
        if (current_value > max_value) {
          max_value = current_value;
        }
      }
      *t_begin = max_value;
    }
  }
  return t;
}

template < typename T >
T min(const T &x, typename T::dim_type d) {
  if (d >= x.dimension()) {
    throw out_of_range("Dimension exceeds limit.");
  }
  typename T::size_storage sz = x.size();
  sz[d] = 1;
  T t(sz);
  if (x.partialContiguity(0, d > 0 ? (d - 1) : 0)
      && x.partialContiguity(d + 1, x.dimension() - 1)) {
    // Get data pointers
    typename T::pointer x_data = x.data();
    typename T::pointer t_data = t.data();
    // Determine left and right size
    typename T::size_type x_left_length = 1;
    for (typename T::dim_type i = 0; i < d; ++i) {
      x_left_length *= x.size(i);
    }
    typename T::size_type x_length = x.size(d);
    typename T::size_type x_right_length = 1;
    for (typename T::dim_type i = d + 1; i < x.dimension(); ++i) {
      x_right_length *= x.size(i);
    }
    // Determine left and right step
    typename T::difference_type x_left_step = d > 0 ? x.stride(d - 1) : 0;
    typename T::difference_type x_step = x.stride(d);
    typename T::difference_type x_right_step = x.stride(x.dimension() - 1);
    typename T::difference_type t_left_step = d > 0 ? t.stride(d - 1) : 0;
    typename T::difference_type t_right_step = t.stride(t.dimension() - 1);
    typename T::value_type min_value =
        ::std::numeric_limits< typename T::value_type >::max();
    typename T::value_type current_value = 0;
    for (typename T::size_type i = 0; i < x_left_length; ++i) {
      for (typename T::size_type j = 0; j < x_right_length; ++j) {
        min_value = ::std::numeric_limits< typename T::value_type >::max();
        for (typename T::size_type k = 0; k < x_length; ++k) {
          current_value =
              x_data[i * x_left_step + j * x_right_step + k * x_step];
          if (current_value < min_value) {
            min_value = current_value;
          }
        }
        t_data[i * t_left_step + j * t_right_step] = min_value;
      }
    }
  } else {
    typename T::size_storage position;
    typename T::size_type size_d = x.size(d);
    typename T::value_type min_value =
        ::std::numeric_limits< typename T::value_type >::max();
    typename T::value_type current_value = 0;
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      min_value = ::std::numeric_limits< typename T::value_type >::max();
      position = t_begin.position();
      for (typename T::size_type i = 0; i < size_d; ++i) {
        position[d] = i;
        current_value = x(position);
        if (current_value < min_value) {
          min_value = current_value;
        }
      }
      *t_begin = min_value;
    }
  }
  return t;
}

template < typename T >
T sum(const T &x, typename T::dim_type d) {
  if (d >= x.dimension()) {
    throw out_of_range("Dimension exceeds limit.");
  }
  typename T::size_storage sz = x.size();
  sz[d] = 1;
  T t(sz);
  if (x.partialContiguity(0, d > 0 ? (d - 1) : 0)
      && x.partialContiguity(d + 1, x.dimension() - 1)) {
    // Get data pointers
    typename T::pointer x_data = x.data();
    typename T::pointer t_data = t.data();
    // Determine left and right size
    typename T::size_type x_left_length = 1;
    for (typename T::dim_type i = 0; i < d; ++i) {
      x_left_length *= x.size(i);
    }
    typename T::size_type x_length = x.size(d);
    typename T::size_type x_right_length = 1;
    for (typename T::dim_type i = d + 1; i < x.dimension(); ++i) {
      x_right_length *= x.size(i);
    }
    // Determine left and right step
    typename T::difference_type x_left_step = d > 0 ? x.stride(d - 1) : 0;
    typename T::difference_type x_step = x.stride(d);
    typename T::difference_type x_right_step = x.stride(x.dimension() - 1);
    typename T::difference_type t_left_step = d > 0 ? t.stride(d - 1) : 0;
    typename T::difference_type t_right_step = t.stride(t.dimension() - 1);
    typename T::value_type sum_value = 0;
    typename T::value_type current_value = 0;
    for (typename T::size_type i = 0; i < x_left_length; ++i) {
      for (typename T::size_type j = 0; j < x_right_length; ++j) {
        sum_value = 0;
        for (typename T::size_type k = 0; k < x_length; ++k) {
          current_value =
              x_data[i * x_left_step + j * x_right_step + k * x_step];
            sum_value += current_value;
        }
        t_data[i * t_left_step + j * t_right_step] = sum_value;
      }
    }
  } else {
    typename T::size_storage position;
    typename T::size_type size_d = x.size(d);
    typename T::value_type sum_value = 0;
    typename T::value_type current_value = 0;
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      sum_value = 0;
      position = t_begin.position();
      for (typename T::size_type i = 0; i < size_d; ++i) {
        position[d] = i;
        current_value = x(position);
        sum_value += current_value;
      }
      *t_begin = sum_value;
    }
  }
  return t;
}

template < typename T >
T prod(const T &x, typename T::dim_type d) {
  if (d >= x.dimension()) {
    throw out_of_range("Dimension exceeds limit.");
  }
  typename T::size_storage sz = x.size();
  sz[d] = 1;
  T t(sz);
  if (x.partialContiguity(0, d > 0 ? (d - 1) : 0)
      && x.partialContiguity(d + 1, x.dimension() - 1)) {
    // Get data pointers
    typename T::pointer x_data = x.data();
    typename T::pointer t_data = t.data();
    // Determine left and right size
    typename T::size_type x_left_length = 1;
    for (typename T::dim_type i = 0; i < d; ++i) {
      x_left_length *= x.size(i);
    }
    typename T::size_type x_length = x.size(d);
    typename T::size_type x_right_length = 1;
    for (typename T::dim_type i = d + 1; i < x.dimension(); ++i) {
      x_right_length *= x.size(i);
    }
    // Determine left and right step
    typename T::difference_type x_left_step = d > 0 ? x.stride(d - 1) : 0;
    typename T::difference_type x_step = x.stride(d);
    typename T::difference_type x_right_step = x.stride(x.dimension() - 1);
    typename T::difference_type t_left_step = d > 0 ? t.stride(d - 1) : 0;
    typename T::difference_type t_right_step = t.stride(t.dimension() - 1);
    typename T::value_type prod_value = 1;
    typename T::value_type current_value = 0;
    for (typename T::size_type i = 0; i < x_left_length; ++i) {
      for (typename T::size_type j = 0; j < x_right_length; ++j) {
        prod_value = 1;
        for (typename T::size_type k = 0; k < x_length; ++k) {
          current_value =
              x_data[i * x_left_step + j * x_right_step + k * x_step];
            prod_value *= current_value;
        }
        t_data[i * t_left_step + j * t_right_step] = prod_value;
      }
    }
  } else {
    typename T::size_storage position;
    typename T::size_type size_d = x.size(d);
    typename T::value_type prod_value = 1;
    typename T::value_type current_value = 0;
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      prod_value = 1;
      position = t_begin.position();
      for (typename T::size_type i = 0; i < size_d; ++i) {
        position[d] = i;
        current_value = x(position);
        prod_value *= current_value;
      }
      *t_begin = prod_value;
    }
  }
  return t;
}

template < typename T >
T mean(const T &x, typename T::dim_type d) {
  if (d >= x.dimension()) {
    throw out_of_range("Dimension exceeds limit.");
  }
  T t = x.sum(d);
  t.div(static_cast< typename T::value_type >(x.size(d)));
  return t;
}

template < typename T >
T var(const T &x, typename T::dim_type d) {
  if (d >= x.dimension()) {
    throw out_of_range("Dimension exceeds limit.");
  }
  T t = x.mean(d);
  if (x.partialContiguity(0, d > 0 ? (d - 1) : 0)
      && x.partialContiguity(d + 1, x.dimension() - 1)) {
    // Get data pointers
    typename T::pointer x_data = x.data();
    typename T::pointer t_data = t.data();
    // Determine left and right size
    typename T::size_type x_left_length = 1;
    for (typename T::dim_type i = 0; i < d; ++i) {
      x_left_length *= x.size(i);
    }
    typename T::size_type x_length = x.size(d);
    typename T::size_type x_right_length = 1;
    for (typename T::dim_type i = d + 1; i < x.dimension(); ++i) {
      x_right_length *= x.size(i);
    }
    // Determine left and right step
    typename T::difference_type x_left_step = d > 0 ? x.stride(d - 1) : 0;
    typename T::difference_type x_step = x.stride(d);
    typename T::difference_type x_right_step = x.stride(x.dimension() - 1);
    typename T::difference_type t_left_step = d > 0 ? t.stride(d - 1) : 0;
    typename T::difference_type t_right_step = t.stride(t.dimension() - 1);
    // Run the loop
    typename T::value_type sum_value = 0;
    typename T::value_type mean_value = 0;
    typename T::value_type current_value = 0;
    for (typename T::size_type i = 0; i < x_left_length; ++i) {
      for (typename T::size_type j = 0; j < x_right_length; ++j) {
        sum_value = 0;
        mean_value = t_data[i * t_left_step + j * t_right_step];
        for (typename T::size_type k = 0; k < x_length; ++k) {
          current_value =
              x_data[i * x_left_step + j * x_right_step + k * x_step];
          sum_value +=
              (current_value - mean_value) * (current_value - mean_value);
        }
        t_data[i * t_left_step + j * t_right_step] =
            sum_value / static_cast< typename T::value_type >(x_length);
      }
    }
  } else {
    typename T::size_storage position;
    typename T::size_type size_d = x.size(d);
    typename T::value_type sum_value = 0;
    typename T::value_type mean_value = 0;
    typename T::value_type current_value = 0;
    for (typename T::reference_iterator t_begin = t.reference_begin(),
             t_end = t.reference_end(); t_begin != t_end; ++t_begin) {
      sum_value = 0;
      mean_value = *t_begin;
      position = t_begin.position();
      for (typename T::size_type i = 0; i < size_d; ++i) {
        position[d] = i;
        current_value = x(position);
        sum_value +=
            (current_value - mean_value) * (current_value - mean_value);
      }
      *t_begin = sum_value / static_cast< typename T::value_type >(size_d);
    }
  }
  return t;
}

template < typename T >
T std(const T &x, typename T::dim_type d) {
  if (d >= x.dimension()) {
    throw out_of_range("Dimension exceeds limit.");
  }
  T t = x.var(d);
  return t.sqrt();
}

}  // namespace math
}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_MATH_INL_REDUCTION_HPP_
