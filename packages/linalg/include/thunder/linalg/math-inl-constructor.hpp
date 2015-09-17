/*
 * \copyright Copyright 2015 Xiang Zhang All Rights Reserved.
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

#ifndef THUNDER_LINALG_MATH_INL_CONSTRUCTOR_HPP_
#define THUNDER_LINALG_MATH_INL_CONSTRUCTOR_HPP_

#include "thunder/linalg/math.hpp"
#include "thunder/linalg/math-inl.hpp"

#include <algorithm>

#include "thunder/exception.hpp"
#include "thunder/tensor.hpp"

namespace thunder {
namespace linalg {
namespace math {

template < typename L >
const typename L::tensor_type& diag(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &r) {
  typedef typename L::tensor_type T;
  typedef tensor::IndexIterator< typename L::size_storage > I;
  if (x.dimension() == 1) {
    // Dimension of x is 1. Construct a diagonal matrix
    if (r.dimension() != 2 || r.size(0) != x.size(1) ||
        r.size(0) != r.size(1)) {
      throw out_of_range("Diag size mismatches.");
    }
    r.zero();
    typename T::pointer x_pointer = x.data();
    typename T::size_type x_size = x.size(0);
    typename T::difference_type x_step = x.stride(0);
    typename T::pointer r_pointer = r.data();
    typename T::difference_type r_step = r.stride(0) + r.stride(1);
    for (typename T::size_type i = 0; i < x_size; ++i) {
      r_pointer[i * r_step] = x_size[i * x_step];
    }
  } else if (x.dimension() == 2) {
    // Dimension of x is 2. Extract diagonal terms.
    if (r.dimension() != 1 || r.size(0) != ::std::min(x.size(0), x.size(1))) {
      throw out_of_range("Diag size mismatches.");
    }
    typename T::pointer x_pointer = x.data();
    typename T::size_type x_size = ::std::min(x.size(0), x.size(1));
    typename T::difference_type x_step = x.stride(0) + x.stride(1);
    typename T::pointer r_pointer = r.data();
    typename T::difference_type r_step = r.stride(0);
    for (typename T::size_type i = 0; i < x_size; ++i) {
      r_pointer[i * r_step] = x_size[i * x_step];
    }
  } else {
    // Dimension of x is 3 or above. Extract diagonal terms in batch mode.
    if (r.dimension() != x.dimension() - 1 || r.size(r.dimension() - 1) !=
        ::std::min(x.size(x.dimension() - 2), x.size(x.dimension() - 1))) {
      throw out_of_range("Diag size mismatches.");
    }
    for (typename T::dim_type i = 0; i < r.dimension() - 1; ++i) {
      if (r.size(i) != x.size(i)) {
        throw out_of_range("Diag size mismatches.");
      }
    }
    typename T::pointer x_pointer = nullptr;
    typename T::size_type x_size =
        ::std::min(x.size(x.dimension() - 2), x.size(x.dimension() - 1));
    typename T::difference_type x_step = x.stride(x.dimension() - 2) +
        x.stride(x.dimension() - 1);
    typename T::pointer r_pointer = nullptr;
    if (x.partialContiguity(0, x.dimension() - 3) &&
        r.partialContiguity(0, r.dimension() - 2)) {
      // Contiguous case
      typename T::size_type batch_size = 1;
      for (typename T::dim_type i = 0; i < r.dimension() - 1; ++i) {
        batch_size = batch_size * r.size(i);
      }
      typename T::difference_type x_batch = x.stride(x.dimension() - 3);
      typename T::difference_type r_batch = r.stride(r.dimension() - 2);
      x_pointer = x.data();
      r_pointer = r.data();
      for (typename T::size_type i = 0; i < batch_size; ++i) {
        for (typename T::size_type j = 0; j < x_size; ++j) {
          r_pointer[i * r_batch + j * r_step] =
              x_pointer[i * x_batch + j * x_step];
        }
      }
    } else {
      // Non-contiguous case
      typename T::size_storage batch_storage(r.size() - 1);
      for (typename T::dim_type i = 0; i < r.dimension() - 1; ++i) {
        batch_storage[i] = r.size(i);
      }
      typename T::difference_type r_step = r.stride(r.dimension() - 1);
      for (I begin = I::begin(batch_storage), end = I::end(batch_storage);
           begin != end; ++begin) {
        x_pointer = x[begin].data();
        r_pointer = r[begin].data();
        for (typename T::size_type i = 0; i < x_size; ++i) {
          r_pointer[i * r_step] = x_pointer[i * x_step];
        }
      }
    }
  }
  return r;
}

template < typename L >
const typename L::tensor_type& eye(
    L *l, const typename L::size_storage &s, const typename L::tensor_type &r) {
  typedef typename L::tensor_type T;
  typedef tensor::IndexIterator< typename L::size_storage > I;
  if (s.size() == 1) {
    if (r.dimension() != 2) {
      throw out_of_range("Eye result dimension mismatches.");
    }
    if (r.size(0) != s[0] || r.size(1) != s[0]) {
      throw out_of_range("Eye result size mismatches.");
    }
  } else {
    if (s.size() != r.dimension()) {
      throw out_of_range("Eye result dimension mismatches.");
    }
    for (typename L::dim_type i = 0; i < s.size(); ++i) {
      if (r.size(i) != s[i]) {
        throw out_of_range("Eye result size mismatches.");
      }
    }
  }

  typename T::size_type rows = r.size(r.dimension() - 2);
  typename T::size_type columns = r.size(r.dimension() - 1);
  typename T::difference_type row_step = r.stride(r.dimension() - 2);
  typename T::difference_type column_step = r.stride(r.dimension() - 1);
  typename T::pointer r_pointer = nullptr;

  if (r.dimension() == 2) {
    r_pointer = r.data();
    for (typename T::size_type i = 0; i < rows; ++i) {
      for (typename T::size_type j = 0; j < columns; ++i) {
        r_pointer[i * row_step + j * column_step] = (i == j ? 1 : 0);
      }
    }
  } else if (r.partialContiguity(0, r.dimension() - 3)) {
    // Contiguous batch mode
    typename T::size_type batch = 1;
    for (typename T::dim_type i = 0; i < r.dimension() - 2; ++i) {
      batch = batch * r.size(i);
    }
    typename T::difference_type batch_step = r.stride(r.dimension() - 3);
    r_pointer = r.data();
    for (typename T::size_type k = 0; k < batch; ++k) {
      for (typename T::size_type i = 0; i < rows; ++i) {
        for (typename T::size_type j = 0; j < columns; ++i) {
          r_pointer[k * batch_step + i * row_step + j * column_step] =
              (i == j ? 1 : 0);
        }
      }
    }
  } else {
    // Non-contiguous batch mode
    typename T::size_storage batch_storage(r.size() - 2);
    for (typename T::dim_type i = 0; i < r.dimension() - 2; ++i) {
      batch_storage[i] = r.size(i);
    }
    for (I begin = I::begin(batch_storage), end = I::end(batch_storage);
         begin != end; ++begin) {
      r_pointer = r[begin].data();
      for (typename T::size_type i = 0; i < rows; ++i) {
        for (typename T::size_type j = 0; j < columns; ++i) {
          r_pointer[i * row_step + j * column_step] = (i == j ? 1 : 0);
        }
      }
    }
  }
  return r;
}

template < typename L >
const typename L::tensor_type& linspace(
    L *l, const typename L::value_type &a, const typename L::value_type &b,
    const typename L::tensor_type &r) {
  typedef typename L::tensor_type T;
  typedef typename T::value_type D;
  if (r.dimension() != 1) {
    throw out_of_range("Linspace result dimension must be 1.");
  }
  typename T::size_type r_size = r.size(0);
  typename T::difference_type r_step = r.stride(0);
  typename T::pointer r_pointer = r.data();
  D divisor = (b - a) / static_cast< D >(r_size - 1);
  r_pointer[0] = a;
  for (typename T::size_type i = 1; i < r_size; ++i) {
    r_pointer[i * r_step] = a + static_cast< D >(i) * divisor;
  }
  return r;
}

template < typename L >
const typename L::tensor_type& logspace(
    L *l, const typename L::value_type &a, const typename L::value_type &b,
    const typename L::tensor_type &r) {
  typedef typename L::tensor_type T;
  typedef typename T::value_type D;
  if (r.dimension() != 1) {
    throw out_of_range("Linspace result dimension must be 1.");
  }
  typename T::size_type r_size = r.size(0);
  typename T::difference_type r_step = r.stride(0);
  typename T::pointer r_pointer = r.data();
  D divisor = (b - a) / static_cast< D >(r_size - 1);
  r_pointer[0] = static_cast< D >(::std::pow(10, a));
  for (typename T::size_type i = 1; i < r_size; ++i) {
    r_pointer[i * r_step] = static_cast< D >(
        ::std::pow(10, a + static_cast< D >(i) * divisor));
  }
  return r;
}

template < typename L >
const typename L::tensor_type& tril(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &r) {
  typedef typename L::tensor_type T;
  typedef tensor::IndexIterator< typename L::size_storage > I;

  if (x.dimension() < 2) {
    throw invalid_argument("Tril input dimension cannot be smaller than 2.");
  }
  if (x.dimension() != r.dimension()) {
    throw out_of_range("Tril result dimension mismatches.");
  }
  if (x.size(x.dimension() - 1) != x.size(x.dimension() - 2)) {
    throw invalid_argument("Tril input must be square matrix.");
  }
  for (typename T::dim_type i = 0; i < x.dimension(); ++i) {
    if (x.size(i) != r.size(i)) {
      throw out_of_range("Tril result size mismatches.");
    }
  }

  typename T::pointer x_pointer = nullptr;
  typename T::pointer r_pointer = nullptr;
  typename T::size_type rows = x.size(x.dimension() - 2);
  typename T::size_type columns = x.size(x.dimension() - 1);
  typename T::difference_type x_row_step = x.stride(x.dimension() - 2);
  typename T::difference_type x_column_step = x.stride(x.dimension() - 1);
  typename T::difference_type r_row_step = r.stride(r.dimension() - 2);
  typename T::difference_type r_column_step = r.stride(r.dimension() - 1);

  if (x.dimension() == 2) {
    x_pointer = x.data();
    r_pointer = r.data();
    for (typename T::size_type i = 0; i < rows; ++i) {
      for (typename T::size_type j = 0; j < columns; ++j) {
        r_pointer[i * r_row_step + j * r_column_step] =
            (i < j ? 0 : x_pointer[i * x_row_step + j * x_column_step]);
      }
    }
  } else if (x.partialContiguity(0, x.dimension() - 3) &&
             r.partialContiguity(0, r.dimension() - 3)) {
    // Contiguous batch
    typename T::size_type batch = 1;
    for (typename T::dim_type i = 0; i < x.dimension() - 2; ++i) {
      batch = batch * x.size(i);
    }
    typename T::difference_type x_batch_step = x.stride(x.dimension() - 3);
    typename T::difference_type r_batch_step = r.stride(r.dimension() - 3);
    x_pointer = x.data();
    r_pointer = r.data();
    for (typename T::size_type k = 0; k < batch; ++k) {
      for (typename T::size_type i = 0; i < rows; ++i) {
        for (typename T::size_type j = 0; j < columns; ++j) {
          r_pointer[k * r_batch_step + i * r_row_step + j * r_column_step] =
              (i < j ? 0 : x_pointer[
                  k * x_batch_step + i * x_row_step + j * x_column_step]);
        }
      }
    }
  } else {
    // Non-contiguous batch
    typename T::size_storage batch_storage(x.dimension() - 2);
    for (typename T::dim_type i = 0; i < x.dimension() - 2; ++i) {
      batch_storage[i] = x.size(i);
    }
    for (I begin = I::begin(batch_storage), end = I::end(batch_storage);
         begin != end; ++begin) {
      x_pointer = x[begin].data();
      r_pointer = r[begin].data();
      for (typename T::size_type i = 0; i < rows; ++i) {
        for (typename T::size_type j = 0; j < columns; ++j) {
          r_pointer[i * r_row_step + j * r_column_step] =
              (i < j ? 0 : x_pointer[i * x_row_step + j * x_column_step]);
        }
      }
    }
  }
  return r;
}

template < typename L >
const typename L::tensor_type& triu(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &r) {
  typedef typename L::tensor_type T;
  typedef tensor::IndexIterator< typename L::size_storage > I;

  if (x.dimension() < 2) {
    throw invalid_argument("Tril input dimension cannot be smaller than 2.");
  }
  if (x.dimension() != r.dimension()) {
    throw out_of_range("Tril result dimension mismatches.");
  }
  if (x.size(x.dimension() - 1) != x.size(x.dimension() - 2)) {
    throw invalid_argument("Tril input must be square matrix.");
  }
  for (typename T::dim_type i = 0; i < x.dimension(); ++i) {
    if (x.size(i) != r.size(i)) {
      throw out_of_range("Tril result size mismatches.");
    }
  }

  typename T::pointer x_pointer = nullptr;
  typename T::pointer r_pointer = nullptr;
  typename T::size_type rows = x.size(x.dimension() - 2);
  typename T::size_type columns = x.size(x.dimension() - 1);
  typename T::difference_type x_row_step = x.stride(x.dimension() - 2);
  typename T::difference_type x_column_step = x.stride(x.dimension() - 1);
  typename T::difference_type r_row_step = r.stride(r.dimension() - 2);
  typename T::difference_type r_column_step = r.stride(r.dimension() - 1);

  if (x.dimension() == 2) {
    x_pointer = x.data();
    r_pointer = r.data();
    for (typename T::size_type i = 0; i < rows; ++i) {
      for (typename T::size_type j = 0; j < columns; ++j) {
        r_pointer[i * r_row_step + j * r_column_step] =
            (i > j ? 0 : x_pointer[i * x_row_step + j * x_column_step]);
      }
    }
  } else if (x.partialContiguity(0, x.dimension() - 3) &&
             r.partialContiguity(0, r.dimension() - 3)) {
    // Contiguous batch
    typename T::size_type batch = 1;
    for (typename T::dim_type i = 0; i < x.dimension() - 2; ++i) {
      batch = batch * x.size(i);
    }
    typename T::difference_type x_batch_step = x.stride(x.dimension() - 3);
    typename T::difference_type r_batch_step = r.stride(r.dimension() - 3);
    x_pointer = x.data();
    r_pointer = r.data();
    for (typename T::size_type k = 0; k < batch; ++k) {
      for (typename T::size_type i = 0; i < rows; ++i) {
        for (typename T::size_type j = 0; j < columns; ++j) {
          r_pointer[k * r_batch_step + i * r_row_step + j * r_column_step] =
              (i > j ? 0 : x_pointer[
                  k * x_batch_step + i * x_row_step + j * x_column_step]);
        }
      }
    }
  } else {
    // Non-contiguous batch
    typename T::size_storage batch_storage(x.dimension() - 2);
    for (typename T::dim_type i = 0; i < x.dimension() - 2; ++i) {
      batch_storage[i] = x.size(i);
    }
    for (I begin = I::begin(batch_storage), end = I::end(batch_storage);
         begin != end; ++begin) {
      x_pointer = x[begin].data();
      r_pointer = r[begin].data();
      for (typename T::size_type i = 0; i < rows; ++i) {
        for (typename T::size_type j = 0; j < columns; ++j) {
          r_pointer[i * r_row_step + j * r_column_step] =
              (i > j ? 0 : x_pointer[i * x_row_step + j * x_column_step]);
        }
      }
    }
  }
  return r;
}

}  // namespace math
}  // namespace linalg
}  // namespace thunder

#endif  // THUNDER_LINALG_MATH_INL_CONSTRUCTOR_HPP_
