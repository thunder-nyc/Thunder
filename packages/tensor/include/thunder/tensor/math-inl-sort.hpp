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

#ifndef THUNDER_TENSOR_MATH_INL_SORT_HPP_
#define THUNDER_TENSOR_MATH_INL_SORT_HPP_

#include "thunder/tensor/math.hpp"
#include "thunder/tensor/math-inl.hpp"

#include <algorithm>

#include "thunder/exception.hpp"

namespace thunder {
namespace tensor {
namespace math {

template < typename D, typename S, typename F >
void quickSort(D *data, S size, F stride) {
  S pivot = 0;
  if (size > 1) {
    // Choose pivot as the middle member
    ::std::swap(data[0], data[size / 2 * stride]);
    // Partition untill end
    for (S i = 1; i < size; ++i) {
      if (data[i * stride] < data[0]) {
        ::std::swap(data[++pivot * stride], data[i * stride]);
      }
    }
    // Swap with where pivot should be
    ::std::swap(data[0], data[pivot * stride]);
    // Sort left side
    quickSort(data, pivot, stride);
    // Sort right side
    quickSort(&data[(pivot + 1) * stride], size - 1 - pivot, stride);
  }
}

template < typename D, typename S, typename F >
void reverseQuickSort(D *data, S size, F stride) {
  S pivot = 0;
  if (size > 1) {
    // Choose pivot as the middle member
    ::std::swap(data[0], data[size / 2 * stride]);
    // Partition untill end
    for (S i = 1; i < size; ++i) {
      if (data[i * stride] > data[0]) {
        ::std::swap(data[++pivot * stride], data[i * stride]);
      }
    }
    // Swap with where pivot should be
    ::std::swap(data[0], data[pivot * stride]);
    // Sort left side
    reverseQuickSort(data, pivot, stride);
    // Sort right side
    reverseQuickSort(&data[(pivot + 1) * stride], size - 1 - pivot, stride);
  }
}

template < typename D, typename S, typename F, typename I, typename G >
void quickSort(D *data, S size, F stride, I *index, G step) {
  S pivot = 0;
  if (size > 1) {
    // Choose pivot as the middle member
    ::std::swap(data[0], data[size / 2 * stride]);
    ::std::swap(index[0], index[size / 2 * step]);
    // Partition untill end
    for (S i = 1; i < size; ++i) {
      if (data[i * stride] < data[0]) {
        ::std::swap(data[++pivot * stride], data[i * stride]);
        ::std::swap(index[pivot * step], index[i * step]);
      }
    }
    // Swap with where pivot should be
    ::std::swap(data[0], data[pivot * stride]);
    ::std::swap(index[0], index[pivot * step]);
    // Sort left side
    quickSort(data, pivot, stride, index, step);
    // Sort right side
    quickSort(&data[(pivot + 1) * stride], size - 1 - pivot, stride,
              &index[(pivot + 1) * step], step);
  }
}

template < typename D, typename S, typename F, typename I, typename G >
void reverseQuickSort(D *data, S size, F stride, I *index, G step) {
  S pivot = 0;
  if (size > 1) {
    // Choose pivot as the middle member
    ::std::swap(data[0], data[size / 2 * stride]);
    ::std::swap(index[0], index[size / 2 * step]);
    // Partition untill end
    for (S i = 1; i < size; ++i) {
      if (data[i * stride] > data[0]) {
        ::std::swap(data[++pivot * stride], data[i * stride]);
        ::std::swap(index[pivot * step], index[i * step]);
      }
    }
    // Swap with where pivot should be
    ::std::swap(data[0], data[pivot * stride]);
    ::std::swap(index[0], index[pivot * step]);
    // Sort left side
    reverseQuickSort(data, pivot, stride, index, step);
    // Sort right side
    reverseQuickSort(&data[(pivot + 1) * stride], size - 1 - pivot, stride,
                     &index[(pivot + 1) * step], step);
  }
}

template < typename T >
const T& sort(const T &x, typename T::dim_type d, bool r) {
  if (d >= x.dimension()) {
    throw out_of_range("Dimension exceeds limit.");
  }

  typename T::difference_type x_step = x.stride(d);
  typename T::size_type x_length = x.size(d);

  if (x.partialContiguity(0, d > 0 ? (d - 1) : d)
      && x.partialContiguity(d + 1, x.dimension() - 1)) {
    // Get data pointers
      typename T::pointer x_pointer = x.data();
    // Determine left and right size
    typename T::size_type x_left_length = 1;
    for (typename T::dim_type i = 0; i < d; ++i) {
      x_left_length *= x.size(i);
    }
    typename T::size_type x_right_length = 1;
    for (typename T::dim_type i = d + 1; i < x.dimension(); ++i) {
      x_right_length *= x.size(i);
    }
    // Determine left and right step
    typename T::difference_type x_left_step = d > 0 ? x.stride(d - 1) : 0;
    typename T::difference_type x_right_step = x.stride(x.dimension() - 1);
    if (r == false) {
      for (typename T::size_type i = 0; i < x_left_length; ++i) {
        for (typename T::size_type j = 0; j < x_right_length; ++j) {
          quickSort(&x_pointer[i * x_left_step + j * x_right_step],
                    x_length, x_step);
        }
      }
    } else {
      for (typename T::size_type i = 0; i < x_left_length; ++i) {
        for (typename T::size_type j = 0; j < x_right_length; ++j) {
          reverseQuickSort(&x_pointer[i * x_left_step + j * x_right_step],
                           x_length, x_step);
        }
      }
    }
  } else {
    T x_narrow = x.narrow(d, 0, 1);
    if (r == false) {
      for (typename T::reference_iterator x_begin =
               x_narrow.reference_begin(), x_end = x_narrow.reference_end();
           x_begin != x_end; ++x_begin) {
        quickSort(&(*x_begin), x_length, x_step);
      }
    } else {
      for (typename T::reference_iterator x_begin =
               x_narrow.reference_begin(), x_end = x_narrow.reference_end();
           x_begin != x_end; ++x_begin) {
        reverseQuickSort(&(*x_begin), x_length, x_step);
      }
    }
  }
  return x;
}

template < typename T >
const T& sort(const T &x, typename T::dim_type d,
              Tensor< typename T::size_storage > *pos, bool r) {
  typedef Tensor< typename T::size_storage > I;
  if (d >= x.dimension()) {
    throw out_of_range("Dimension exceeds limit.");
  }

  typename T::difference_type x_step = x.stride(d);
  typename T::size_type x_length = x.size(d);

  // Initialize pos
  pos->resizeAs(x);
  for (typename T::size_type i = 0; i < x_length; ++i) {
    pos->select(d, i).fill(static_cast<typename I::value_type >(i));
  }
  typename T::difference_type p_step = pos->stride(d);

  if (x.partialContiguity(0, d > 0 ? (d - 1) : 0)
      && x.partialContiguity(d + 1, x.dimension() - 1)) {
    // Get data pointers
    typename T::pointer x_pointer = x.data();
    typename I::pointer p_pointer = pos->data();
    // Determine left and right size
    typename T::size_type x_left_length = 1;
    for (typename T::dim_type i = 0; i < d; ++i) {
      x_left_length *= x.size(i);
    }
    typename T::size_type x_right_length = 1;
    for (typename T::dim_type i = d + 1; i < x.dimension(); ++i) {
      x_right_length *= x.size(i);
    }
    // Determine left and right step
    typename T::difference_type x_left_step = d > 0 ? x.stride(d - 1) : 0;
    typename I::difference_type p_left_step = d > 0 ? pos->stride(d - 1) : 0;
    typename T::difference_type x_right_step = x.stride(x.dimension() - 1);
    typename I::difference_type p_right_step =
        pos->stride(pos->dimension() - 1);
    if (r == false) {
      for (typename T::size_type i = 0; i < x_left_length; ++i) {
        for (typename T::size_type j = 0; j < x_right_length; ++j) {
          quickSort(
              &x_pointer[i * x_left_step + j * x_right_step], x_length, x_step,
              &p_pointer[i * p_left_step + j * p_right_step], p_step);
        }
      }
    } else {
      for (typename T::size_type i = 0; i < x_left_length; ++i) {
        for (typename T::size_type j = 0; j < x_right_length; ++j) {
          reverseQuickSort(
              &x_pointer[i * x_left_step + j * x_right_step], x_length, x_step,
              &p_pointer[i * p_left_step + j * p_right_step], p_step);
        }
      }
    }
  } else {
    T x_narrow = x.narrow(d, 0, 1);
    I p_narrow = pos->narrow(d, 0, 1);
    typename I::reference_iterator p_begin = p_narrow.reference_begin();
    if (r == false) {
      for (typename T::reference_iterator x_begin =
               x_narrow.reference_begin(), x_end = x_narrow.reference_end();
           x_begin != x_end; ++x_begin, ++p_begin) {
        quickSort(
            &(*x_begin), x_length, x_step, &(*p_begin), p_step);
      }
    } else {
      for (typename T::reference_iterator x_begin =
               x_narrow.reference_begin(), x_end = x_narrow.reference_end();
           x_begin != x_end; ++x_begin, ++p_begin) {
        reverseQuickSort(
            &(*x_begin), x_length, x_step, &(*p_begin), p_step);
      }
    }
  }

  return x;
}

}  // namespace math
}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_MATH_INL_SORT_HPP_
