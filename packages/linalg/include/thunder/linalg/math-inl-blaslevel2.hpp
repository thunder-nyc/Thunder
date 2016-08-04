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

#ifndef THUNDER_LINALG_MATH_INL_BLASLEVEL2_HPP_
#define THUNDER_LINALG_MATH_INL_BLASLEVEL2_HPP_

#include "thunder/linalg/math.hpp"
#include "thunder/linalg/math-inl.hpp"

#include <cmath>
#include <complex>

#include "thunder/exception.hpp"
#include "thunder/linalg/cxxblas.hpp"
#include "thunder/tensor.hpp"

namespace thunder {
namespace linalg {
namespace math {

template < typename L >
const typename L::tensor_type& gbmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::size_type kl,
    typename L::size_type ku) {
  typedef typename L::tensor_type T;
  if (a.dimension() != x.dimension() + 1 ||
      a.dimension() != y.dimension() + 1) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (a.size(i) != x.size(i) || a.size(i) != y.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.size(a.dimension() - 2) != y.size(y.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.size(a.dimension() - 1) != kl + ku + 1) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous");
  }

  int m = static_cast< int >(a.size(a.dimension() - 2));
  int n = static_cast< int >(x.size(x.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    cxxblas::gbmv(m, n, a_pointer, x_pointer, y_pointer, alpha, beta, kl, ku,
                  a_step, x_step, y_step);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             x.partialContiguity(0, x.dimension() - 2) &&
             y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::gbmv(m, n, &a_pointer[i * a_batch], &x_pointer[i * x_batch],
                    &y_pointer[i * y_batch], alpha, beta, kl, ku, a_step,
                    x_step, y_step);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::gbmv(m, n, a[x_begin.position()].data(), &(*x_begin),
                    y[x_begin.position()].data(), alpha, beta, kl, ku, a_step,
                    x_step, y_step);
    }
  }
  return y;
}

template < typename L >
const typename L::tensor_type& gemv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta) {
  typedef typename L::tensor_type T;
  if (a.dimension() != x.dimension() + 1 ||
      a.dimension() != y.dimension() + 1) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (a.size(i) != x.size(i) || a.size(i) != y.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.size(a.dimension() - 1) != x.size(x.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.size(a.dimension() - 2) != y.size(y.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous");
  }

  int m = static_cast< int >(a.size(a.dimension() - 2));
  int n = static_cast< int >(a.size(a.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    cxxblas::gemv(m, n, a_pointer, x_pointer, y_pointer, alpha, beta, a_step,
                  x_step, y_step);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             x.partialContiguity(0, x.dimension() - 2) &&
             y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::gemv(m, n, &a_pointer[i * a_batch], &x_pointer[i * x_batch],
                    &y_pointer[i * y_batch], alpha, beta, a_step, x_step,
                    y_step);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::gemv(m, n, a[x_begin.position()].data(), &(*x_begin),
                    y[x_begin.position()].data(), alpha, beta, a_step, x_step,
                    y_step);
    }
  }

  return y;
}

template < typename L >
const typename L::tensor_type& ger(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &a, const typename L::value_type &alpha) {
  typedef typename L::tensor_type T;
  if (a.dimension() != x.dimension() + 1 ||
      a.dimension() != y.dimension() + 1) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (a.size(i) != x.size(i) || a.size(i) != y.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.size(a.dimension() - 2) != x.size(x.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.size(a.dimension() - 1) != y.size(y.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous");
  }

  int m = static_cast< int >(x.size(x.dimension() - 1));
  int n = static_cast< int >(y.size(y.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    cxxblas::ger(m, n, x_pointer, y_pointer, a_pointer, alpha, x_step, y_step,
                 a_step);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             x.partialContiguity(0, x.dimension() - 2) &&
             y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::ger(m, n, &x_pointer[i * x_batch], &y_pointer[i * y_batch],
                    &a_pointer[i * a_batch], alpha, x_step, y_step, a_step);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::ger(m, n, &(*x_begin), y[x_begin.position()].data(),
                    a[x_begin.position()].data(), alpha, x_step, y_step,
                    a_step);
    }
  }
  return a;
}

template < typename L >
const typename L::tensor_type& gerc(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &a, const typename L::value_type &alpha) {
  typedef typename L::tensor_type T;
  if (a.dimension() != x.dimension() + 1 ||
      a.dimension() != y.dimension() + 1) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (a.size(i) != x.size(i) || a.size(i) != y.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.size(a.dimension() - 2) != x.size(x.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.size(a.dimension() - 1) != y.size(y.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous");
  }

  int m = static_cast< int >(x.size(x.dimension() - 1));
  int n = static_cast< int >(y.size(y.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    cxxblas::gerc(m, n, x_pointer, y_pointer, a_pointer, alpha, x_step, y_step,
                  a_step);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             x.partialContiguity(0, x.dimension() - 2) &&
             y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::gerc(m, n, &x_pointer[i * x_batch], &y_pointer[i * y_batch],
                    &a_pointer[i * a_batch], alpha, x_step, y_step, a_step);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::gerc(m, n, &(*x_begin), y[x_begin.position()].data(),
                    a[x_begin.position()].data(), alpha, x_step, y_step,
                    a_step);
    }
  }
  return a;
}

template < typename L >
const typename L::tensor_type& hbmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::size_type k,
    typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (a.dimension() != x.dimension() + 1 ||
      a.dimension() != y.dimension() + 1) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (a.size(i) != x.size(i) || a.size(i) != y.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.size(a.dimension() - 2) != y.size(y.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.size(a.dimension() - 1) != k + 1) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous");
  }

  int n = static_cast< int >(a.size(a.dimension() - 2));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    cxxblas::hbmv(n, a_pointer, x_pointer, y_pointer, alpha, beta, k, a_step,
                  x_step, y_step, cxxblas::Order::kRowMajor, uplo);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             x.partialContiguity(0, x.dimension() - 2) &&
             y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::hbmv(n, &a_pointer[i * a_batch], &x_pointer[i * x_batch],
                    &y_pointer[i * y_batch], alpha, beta, k, a_step, x_step,
                    y_step, cxxblas::Order::kRowMajor, uplo);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::hbmv(n, a[x_begin.position()].data(), &(*x_begin),
                    y[x_begin.position()].data(), alpha, beta, k, a_step,
                    x_step, y_step, cxxblas::Order::kRowMajor, uplo);
    }
  }
  return y;
}

template < typename L >
const typename L::tensor_type& hemv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (a.dimension() != x.dimension() + 1 ||
      a.dimension() != y.dimension() + 1) {
    throw out_of_range("Tensor dimension does not match.");
  }
  if (a.size(a.dimension() - 2) != a.size(a.dimension() - 1)) {
    throw out_of_range("Matrix size is not symmetric.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (a.size(i) != x.size(i) || a.size(i) != y.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.size(a.dimension() - 1) != x.size(x.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.size(a.dimension() - 2) != y.size(y.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1) {
    throw contiguity_error("Matrix last dimension is not contiguous");
  }

  int n = static_cast< int >(a.size(a.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    cxxblas::hemv(n, a_pointer, x_pointer, y_pointer, alpha, beta, a_step,
                  x_step, y_step, cxxblas::Order::kRowMajor, uplo);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             x.partialContiguity(0, x.dimension() - 2) &&
             y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::hemv(n, &a_pointer[i * a_batch], &x_pointer[i * x_batch],
                    &y_pointer[i * y_batch], alpha, beta, a_step, x_step,
                    y_step, cxxblas::Order::kRowMajor, uplo);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::hemv(n, a[x_begin.position()].data(), &(*x_begin),
                    y[x_begin.position()].data(), alpha, beta, a_step, x_step,
                    y_step, cxxblas::Order::kRowMajor, uplo);
    }
  }

  return y;
}

template < typename L >
const typename L::tensor_type& her(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &a,
    const typename L::real_type &alpha, typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (a.dimension() != x.dimension() + 1) {
    throw out_of_range("Tensor dimension does not match.");
  }
  if (a.size(a.dimension() - 2) != a.size(a.dimension() - 1)) {
    throw out_of_range("Matrix size is not symmetric.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (a.size(i) != x.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.size(a.dimension() - 1) != x.size(x.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1) {
    throw contiguity_error("Matrix last dimension is not contiguous");
  }

  int n = static_cast< int >(a.size(a.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer x_pointer = x.data();
    cxxblas::her(n, x_pointer, a_pointer, alpha, x_step, a_step,
                 cxxblas::Order::kRowMajor, uplo);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             x.partialContiguity(0, x.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::her(n, &x_pointer[i * x_batch], &a_pointer[i * a_batch],
                   alpha, x_step, a_step,
                   cxxblas::Order::kRowMajor, uplo);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::her(n, &(*x_begin), a[x_begin.position()].data(),
                   alpha, x_step, a_step,
                   cxxblas::Order::kRowMajor, uplo);
    }
  }

  return a;
}

template < typename L >
const typename L::tensor_type& her2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &a, const typename L::value_type &alpha,
    typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (a.dimension() != x.dimension() + 1 ||
      a.dimension() != y.dimension() + 1) {
    throw out_of_range("Tensor dimension does not match.");
  }
  if (a.size(a.dimension() - 2) != a.size(a.dimension() - 1)) {
    throw out_of_range("Matrix size is not symmetric.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (a.size(i) != x.size(i) || a.size(i) != y.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.size(a.dimension() - 2) != x.size(x.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.size(a.dimension() - 1) != y.size(y.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous");
  }

  int n = static_cast< int >(a.size(a.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    cxxblas::her2(n, x_pointer, y_pointer, a_pointer, alpha, x_step, y_step,
                  a_step, cxxblas::Order::kRowMajor, uplo);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             x.partialContiguity(0, x.dimension() - 2) &&
             y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::her2(n, &x_pointer[i * x_batch], &y_pointer[i * y_batch],
                    &a_pointer[i * a_batch], alpha, x_step, y_step, a_step,
                    cxxblas::Order::kRowMajor, uplo);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::her2(n, &(*x_begin), y[x_begin.position()].data(),
                    a[x_begin.position()].data(), alpha, x_step, y_step,
                    a_step, cxxblas::Order::kRowMajor, uplo);
    }
  }
  return a;
}

template < typename L >
const typename L::tensor_type& hpmv(
    L *l, const typename L::tensor_type &ap, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (ap.dimension() != x.dimension() ||
      ap.dimension() != y.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  if (ap.size(ap.dimension() - 1) !=
      ((x.size(x.dimension() - 1) + 1) * x.size(x.dimension() - 1) / 2)) {
    throw out_of_range("Tensor size does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (ap.size(i) != x.size(i) || ap.size(i) != y.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (x.size(x.dimension() - 1) != y.size(y.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (ap.stride(ap.dimension() - 1) != 1) {
    throw contiguity_error("Matrix last dimension is not contiguous");
  }

  int n = static_cast< int >(x.size(x.dimension() - 1));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));

  if (ap.dimension() == 1) {
    typename T::pointer ap_pointer = ap.data();
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    cxxblas::hpmv(n, ap_pointer, x_pointer, y_pointer, alpha, beta, x_step,
                  y_step, cxxblas::Order::kRowMajor, uplo);
  } else if (ap.partialContiguity(0, ap.dimension() - 2) &&
             x.partialContiguity(0, x.dimension() - 2) &&
             y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < ap.dimension() - 1; ++i) {
      batch_size = batch_size * ap.size(i);
    }
    typename T::pointer ap_pointer = ap.data();
    typename T::difference_type ap_batch = ap.stride(ap.dimension() - 2);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::hpmv(n, &ap_pointer[i * ap_batch], &x_pointer[i * x_batch],
                    &y_pointer[i * y_batch], alpha, beta, x_step, y_step,
                    cxxblas::Order::kRowMajor, uplo);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::hpmv(n, ap[x_begin.position()].data(), &(*x_begin),
                    y[x_begin.position()].data(), alpha, beta, x_step, y_step,
                    cxxblas::Order::kRowMajor, uplo);
    }
  }

  return y;
}

template < typename L >
const typename L::tensor_type& hpr(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &ap,
    const typename L::real_type &alpha, typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (ap.dimension() != x.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  if (ap.size(ap.dimension() - 1) !=
      ((x.size(x.dimension() - 1) + 1) * x.size(x.dimension() - 1) / 2)) {
    throw out_of_range("Tensor size does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (ap.size(i) != x.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (ap.stride(ap.dimension() - 1) != 1) {
    throw contiguity_error("Matrix last dimension is not contiguous");
  }

  int n = static_cast< int >(x.size(x.dimension() - 1));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));

  if (ap.dimension() == 1) {
    typename T::pointer ap_pointer = ap.data();
    typename T::pointer x_pointer = x.data();
    cxxblas::hpr(n, x_pointer, ap_pointer, alpha, x_step,
                 cxxblas::Order::kRowMajor, uplo);
  } else if (ap.partialContiguity(0, ap.dimension() - 2) &&
             x.partialContiguity(0, x.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < ap.dimension() - 1; ++i) {
      batch_size = batch_size * ap.size(i);
    }
    typename T::pointer ap_pointer = ap.data();
    typename T::difference_type ap_batch = ap.stride(ap.dimension() - 2);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::hpr(n, &x_pointer[i * x_batch], &ap_pointer[i * ap_batch],
                   alpha, x_step, cxxblas::Order::kRowMajor, uplo);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::hpr(n, &(*x_begin), ap[x_begin.position()].data(),
                   alpha, x_step, cxxblas::Order::kRowMajor, uplo);
    }
  }

  return ap;
}

template < typename L >
const typename L::tensor_type& hpr2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &ap, const typename L::value_type &alpha,
    typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (ap.dimension() != x.dimension() ||
      ap.dimension() != y.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  if (ap.size(ap.dimension() - 1) !=
      ((x.size(x.dimension() - 1) + 1) * x.size(x.dimension() - 1) / 2)) {
    throw out_of_range("Tensor size does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (ap.size(i) != x.size(i) || ap.size(i) != y.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (x.size(x.dimension() - 1) != y.size(y.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (ap.stride(ap.dimension() - 1) != 1) {
    throw contiguity_error("Matrix last dimension is not contiguous");
  }

  int n = static_cast< int >(x.size(x.dimension() - 1));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));

  if (ap.dimension() == 1) {
    typename T::pointer ap_pointer = ap.data();
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    cxxblas::hpr2(n, x_pointer, y_pointer, ap_pointer, alpha, x_step, y_step,
                  cxxblas::Order::kRowMajor, uplo);
  } else if (ap.partialContiguity(0, ap.dimension() - 2) &&
             x.partialContiguity(0, x.dimension() - 2) &&
             y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < ap.dimension() - 1; ++i) {
      batch_size = batch_size * ap.size(i);
    }
    typename T::pointer ap_pointer = ap.data();
    typename T::difference_type ap_batch = ap.stride(ap.dimension() - 2);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::hpr2(n, &x_pointer[i * x_batch], &y_pointer[i * y_batch],
                    &ap_pointer[i * ap_batch], alpha, x_step, y_step,
                    cxxblas::Order::kRowMajor, uplo);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::hpr2(n, &(*x_begin), y[x_begin.position()].data(),
                    ap[x_begin.position()].data(), alpha, x_step, y_step,
                    cxxblas::Order::kRowMajor, uplo);
    }
  }

  return y;
}

template < typename L >
const typename L::tensor_type& sbmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::size_type k,
    typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (a.dimension() != x.dimension() + 1 ||
      a.dimension() != y.dimension() + 1) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (a.size(i) != x.size(i) || a.size(i) != y.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.size(a.dimension() - 2) != y.size(y.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.size(a.dimension() - 1) != k + 1) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous");
  }

  int n = static_cast< int >(a.size(a.dimension() - 2));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    cxxblas::sbmv(n, a_pointer, x_pointer, y_pointer, alpha, beta, k, a_step,
                  x_step, y_step, cxxblas::Order::kRowMajor, uplo);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             x.partialContiguity(0, x.dimension() - 2) &&
             y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::sbmv(n, &a_pointer[i * a_batch], &x_pointer[i * x_batch],
                    &y_pointer[i * y_batch], alpha, beta, k, a_step, x_step,
                    y_step, cxxblas::Order::kRowMajor, uplo);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::sbmv(n, a[x_begin.position()].data(), &(*x_begin),
                    y[x_begin.position()].data(), alpha, beta, k, a_step,
                    x_step, y_step, cxxblas::Order::kRowMajor, uplo);
    }
  }
  return y;
}

template < typename L >
const typename L::tensor_type& spmv(
    L *l, const typename L::tensor_type &ap, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (ap.dimension() != x.dimension() ||
      ap.dimension() != y.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  if (ap.size(ap.dimension() - 1) !=
      ((x.size(x.dimension() - 1) + 1) * x.size(x.dimension() - 1) / 2)) {
    throw out_of_range("Tensor size does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (ap.size(i) != x.size(i) || ap.size(i) != y.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (x.size(x.dimension() - 1) != y.size(y.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (ap.stride(ap.dimension() - 1) != 1) {
    throw contiguity_error("Matrix last dimension is not contiguous");
  }

  int n = static_cast< int >(x.size(x.dimension() - 1));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));

  if (ap.dimension() == 1) {
    typename T::pointer ap_pointer = ap.data();
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    cxxblas::spmv(n, ap_pointer, x_pointer, y_pointer, alpha, beta, x_step,
                  y_step, cxxblas::Order::kRowMajor, uplo);
  } else if (ap.partialContiguity(0, ap.dimension() - 2) &&
             x.partialContiguity(0, x.dimension() - 2) &&
             y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < ap.dimension() - 1; ++i) {
      batch_size = batch_size * ap.size(i);
    }
    typename T::pointer ap_pointer = ap.data();
    typename T::difference_type ap_batch = ap.stride(ap.dimension() - 2);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::spmv(n, &ap_pointer[i * ap_batch], &x_pointer[i * x_batch],
                    &y_pointer[i * y_batch], alpha, beta, x_step, y_step,
                    cxxblas::Order::kRowMajor, uplo);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::spmv(n, ap[x_begin.position()].data(), &(*x_begin),
                    y[x_begin.position()].data(), alpha, beta, x_step, y_step,
                    cxxblas::Order::kRowMajor, uplo);
    }
  }

  return y;
}

template < typename L >
const typename L::tensor_type& spr(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &ap,
    const typename L::real_type &alpha, typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (ap.dimension() != x.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  if (ap.size(ap.dimension() - 1) !=
      ((x.size(x.dimension() - 1) + 1) * x.size(x.dimension() - 1) / 2)) {
    throw out_of_range("Tensor size does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (ap.size(i) != x.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (ap.stride(ap.dimension() - 1) != 1) {
    throw contiguity_error("Matrix last dimension is not contiguous");
  }

  int n = static_cast< int >(x.size(x.dimension() - 1));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));

  if (ap.dimension() == 1) {
    typename T::pointer ap_pointer = ap.data();
    typename T::pointer x_pointer = x.data();
    cxxblas::spr(n, x_pointer, ap_pointer, alpha, x_step,
                 cxxblas::Order::kRowMajor, uplo);
  } else if (ap.partialContiguity(0, ap.dimension() - 2) &&
             x.partialContiguity(0, x.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < ap.dimension() - 1; ++i) {
      batch_size = batch_size * ap.size(i);
    }
    typename T::pointer ap_pointer = ap.data();
    typename T::difference_type ap_batch = ap.stride(ap.dimension() - 2);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::spr(n, &x_pointer[i * x_batch], &ap_pointer[i * ap_batch],
                   alpha, x_step, cxxblas::Order::kRowMajor, uplo);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::spr(n, &(*x_begin), ap[x_begin.position()].data(),
                   alpha, x_step, cxxblas::Order::kRowMajor, uplo);
    }
  }

  return ap;
}

template < typename L >
const typename L::tensor_type& spr2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &ap, const typename L::value_type &alpha,
    typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (ap.dimension() != x.dimension() ||
      ap.dimension() != y.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  if (ap.size(ap.dimension() - 1) !=
      ((x.size(x.dimension() - 1) + 1) * x.size(x.dimension() - 1) / 2)) {
    throw out_of_range("Tensor size does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (ap.size(i) != x.size(i) || ap.size(i) != y.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (x.size(x.dimension() - 1) != y.size(y.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (ap.stride(ap.dimension() - 1) != 1) {
    throw contiguity_error("Matrix last dimension is not contiguous");
  }

  int n = static_cast< int >(x.size(x.dimension() - 1));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));

  if (ap.dimension() == 1) {
    typename T::pointer ap_pointer = ap.data();
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    cxxblas::spr2(n, x_pointer, y_pointer, ap_pointer, alpha, x_step, y_step,
                  cxxblas::Order::kRowMajor, uplo);
  } else if (ap.partialContiguity(0, ap.dimension() - 2) &&
             x.partialContiguity(0, x.dimension() - 2) &&
             y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < ap.dimension() - 1; ++i) {
      batch_size = batch_size * ap.size(i);
    }
    typename T::pointer ap_pointer = ap.data();
    typename T::difference_type ap_batch = ap.stride(ap.dimension() - 2);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::spr2(n, &x_pointer[i * x_batch], &y_pointer[i * y_batch],
                    &ap_pointer[i * ap_batch], alpha, x_step, y_step,
                    cxxblas::Order::kRowMajor, uplo);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::spr2(n, &(*x_begin), y[x_begin.position()].data(),
                    ap[x_begin.position()].data(), alpha, x_step, y_step,
                    cxxblas::Order::kRowMajor, uplo);
    }
  }

  return y;
}

template < typename L >
const typename L::tensor_type& symv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    const typename L::tensor_type &y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (a.dimension() != x.dimension() + 1 ||
      a.dimension() != y.dimension() + 1) {
    throw out_of_range("Tensor dimension does not match.");
  }
  if (a.size(a.dimension() - 2) != a.size(a.dimension() - 1)) {
    throw out_of_range("Matrix size is not symmetric.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (a.size(i) != x.size(i) || a.size(i) != y.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.size(a.dimension() - 1) != x.size(x.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.size(a.dimension() - 2) != y.size(y.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1) {
    throw contiguity_error("Matrix last dimension is not contiguous");
  }

  int n = static_cast< int >(a.size(a.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    cxxblas::symv(n, a_pointer, x_pointer, y_pointer, alpha, beta, a_step,
                  x_step, y_step, cxxblas::Order::kRowMajor, uplo);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             x.partialContiguity(0, x.dimension() - 2) &&
             y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::symv(n, &a_pointer[i * a_batch], &x_pointer[i * x_batch],
                    &y_pointer[i * y_batch], alpha, beta, a_step, x_step,
                    y_step, cxxblas::Order::kRowMajor, uplo);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::symv(n, a[x_begin.position()].data(), &(*x_begin),
                    y[x_begin.position()].data(), alpha, beta, a_step, x_step,
                    y_step, cxxblas::Order::kRowMajor, uplo);
    }
  }

  return y;
}

template < typename L >
const typename L::tensor_type& syr(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &a,
    const typename L::real_type &alpha, typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (a.dimension() != x.dimension() + 1) {
    throw out_of_range("Tensor dimension does not match.");
  }
  if (a.size(a.dimension() - 2) != a.size(a.dimension() - 1)) {
    throw out_of_range("Matrix size is not symmetric.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (a.size(i) != x.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.size(a.dimension() - 1) != x.size(x.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1) {
    throw contiguity_error("Matrix last dimension is not contiguous");
  }

  int n = static_cast< int >(a.size(a.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer x_pointer = x.data();
    cxxblas::syr(n, x_pointer, a_pointer, alpha, x_step, a_step,
                 cxxblas::Order::kRowMajor, uplo);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             x.partialContiguity(0, x.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::syr(n, &x_pointer[i * x_batch], &a_pointer[i * a_batch],
                   alpha, x_step, a_step, cxxblas::Order::kRowMajor, uplo);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::syr(n, &(*x_begin), a[x_begin.position()].data(),
                   alpha, x_step, a_step, cxxblas::Order::kRowMajor, uplo);
    }
  }

  return a;
}

template < typename L >
const typename L::tensor_type& syr2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &a, const typename L::value_type &alpha,
    typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (a.dimension() != x.dimension() + 1 ||
      a.dimension() != y.dimension() + 1) {
    throw out_of_range("Tensor dimension does not match.");
  }
  if (a.size(a.dimension() - 2) != a.size(a.dimension() - 1)) {
    throw out_of_range("Matrix size is not symmetric.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (a.size(i) != x.size(i) || a.size(i) != y.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.size(a.dimension() - 2) != x.size(x.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.size(a.dimension() - 1) != y.size(y.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous");
  }

  int n = static_cast< int >(a.size(a.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    cxxblas::syr2(n, x_pointer, y_pointer, a_pointer, alpha, x_step, y_step,
                  a_step, cxxblas::Order::kRowMajor, uplo);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             x.partialContiguity(0, x.dimension() - 2) &&
             y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::syr2(n, &x_pointer[i * x_batch], &y_pointer[i * y_batch],
                    &a_pointer[i * a_batch], alpha, x_step, y_step, a_step,
                    cxxblas::Order::kRowMajor, uplo);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::syr2(n, &(*x_begin), y[x_begin.position()].data(),
                    a[x_begin.position()].data(), alpha, x_step, y_step,
                    a_step, cxxblas::Order::kRowMajor, uplo);
    }
  }
  return a;
}

template < typename L >
const typename L::tensor_type& tbmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::size_type k, typename L::Uplo uplo,
    typename L::Diag diag) {
  typedef typename L::tensor_type T;
  if (a.dimension() != x.dimension() + 1) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (a.size(i) != x.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.size(a.dimension() - 2) != x.size(x.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.size(a.dimension() - 1) != k + 1) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous");
  }

  int n = static_cast< int >(x.size(x.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer x_pointer = x.data();
    cxxblas::tbmv(n, a_pointer, x_pointer, k, a_step, x_step,
                  cxxblas::Order::kRowMajor, uplo, cxxblas::Trans::kNoTrans,
                  diag);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             x.partialContiguity(0, x.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::tbmv(n, &a_pointer[i * a_batch], &x_pointer[i * x_batch],
                    k, a_step, x_step, cxxblas::Order::kRowMajor, uplo,
                    cxxblas::Trans::kNoTrans, diag);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::tbmv(n, a[x_begin.position()].data(), &(*x_begin),
                    k, a_step, x_step, cxxblas::Order::kRowMajor, uplo,
                    cxxblas::Trans::kNoTrans, diag);
    }
  }
  return x;
}

template < typename L >
const typename L::tensor_type& tbsv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::size_type k, typename L::Uplo uplo,
    typename L::Diag diag) {
  typedef typename L::tensor_type T;
  if (a.dimension() != x.dimension() + 1) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (a.size(i) != x.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.size(a.dimension() - 2) != x.size(x.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.size(a.dimension() - 1) != k + 1) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous");
  }

  int n = static_cast< int >(x.size(x.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer x_pointer = x.data();
    cxxblas::tbsv(n, a_pointer, x_pointer, k, a_step, x_step,
                  cxxblas::Order::kRowMajor, uplo, cxxblas::Trans::kNoTrans,
                  diag);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             x.partialContiguity(0, x.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::tbsv(n, &a_pointer[i * a_batch], &x_pointer[i * x_batch],
                    k, a_step, x_step, cxxblas::Order::kRowMajor, uplo,
                    cxxblas::Trans::kNoTrans, diag);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::tbsv(n, a[x_begin.position()].data(), &(*x_begin),
                    k, a_step, x_step, cxxblas::Order::kRowMajor, uplo,
                    cxxblas::Trans::kNoTrans, diag);
    }
  }
  return x;
}

template < typename L >
const typename L::tensor_type& tpmv(
    L *l, const typename L::tensor_type &ap, const typename L::tensor_type &x,
    typename L::Uplo uplo, typename L::Diag diag) {
  typedef typename L::tensor_type T;
  if (ap.dimension() != x.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  if (ap.size(ap.dimension() - 1) !=
      ((x.size(x.dimension() - 1) + 1) * x.size(x.dimension() - 1) / 2)) {
    throw out_of_range("Tensor size does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (ap.size(i) != x.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (ap.stride(ap.dimension() - 1) != 1) {
    throw contiguity_error("Matrix last dimension is not contiguous");
  }

  int n = static_cast< int >(x.size(x.dimension() - 1));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));

  if (ap.dimension() == 1) {
    typename T::pointer ap_pointer = ap.data();
    typename T::pointer x_pointer = x.data();
    cxxblas::tpmv(n, ap_pointer, x_pointer, x_step, cxxblas::Order::kRowMajor,
                  uplo, cxxblas::Trans::kNoTrans, diag);
  } else if (ap.partialContiguity(0, ap.dimension() - 2) &&
             x.partialContiguity(0, x.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < ap.dimension() - 1; ++i) {
      batch_size = batch_size * ap.size(i);
    }
    typename T::pointer ap_pointer = ap.data();
    typename T::difference_type ap_batch = ap.stride(ap.dimension() - 2);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::tpmv(n, &ap_pointer[i * ap_batch], &x_pointer[i * x_batch],
                    x_step, cxxblas::Order::kRowMajor, uplo,
                    cxxblas::Trans::kNoTrans, diag);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::tpmv(n, ap[x_begin.position()].data(), &(*x_begin),
                    x_step, cxxblas::Order::kRowMajor, uplo,
                    cxxblas::Trans::kNoTrans, diag);
    }
  }

  return x;
}

template < typename L >
const typename L::tensor_type& tpsv(
    L *l, const typename L::tensor_type &ap, const typename L::tensor_type &x,
    typename L::Uplo uplo, typename L::Diag diag) {
  typedef typename L::tensor_type T;
  if (ap.dimension() != x.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  if (ap.size(ap.dimension() - 1) !=
      ((x.size(x.dimension() - 1) + 1) * x.size(x.dimension() - 1) / 2)) {
    throw out_of_range("Tensor size does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (ap.size(i) != x.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (ap.stride(ap.dimension() - 1) != 1) {
    throw contiguity_error("Matrix last dimension is not contiguous");
  }

  int n = static_cast< int >(x.size(x.dimension() - 1));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));

  if (ap.dimension() == 1) {
    typename T::pointer ap_pointer = ap.data();
    typename T::pointer x_pointer = x.data();
    cxxblas::tpsv(n, ap_pointer, x_pointer, x_step, cxxblas::Order::kRowMajor,
                  uplo, cxxblas::Trans::kNoTrans, diag);
  } else if (ap.partialContiguity(0, ap.dimension() - 2) &&
             x.partialContiguity(0, x.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < ap.dimension() - 1; ++i) {
      batch_size = batch_size * ap.size(i);
    }
    typename T::pointer ap_pointer = ap.data();
    typename T::difference_type ap_batch = ap.stride(ap.dimension() - 2);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::tpsv(n, &ap_pointer[i * ap_batch], &x_pointer[i * x_batch],
                    x_step, cxxblas::Order::kRowMajor, uplo,
                    cxxblas::Trans::kNoTrans, diag);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::tpsv(n, ap[x_begin.position()].data(), &(*x_begin),
                    x_step, cxxblas::Order::kRowMajor, uplo,
                    cxxblas::Trans::kNoTrans, diag);
    }
  }

  return x;
}

template < typename L >
const typename L::tensor_type& trmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::Uplo uplo, typename L::Diag diag) {
  typedef typename L::tensor_type T;
  if (a.dimension() != x.dimension() + 1) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (a.size(i) != x.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.size(a.dimension() - 1) != x.size(x.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.size(a.dimension() - 2) != x.size(x.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous");
  }

  int n = static_cast< int >(a.size(a.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer x_pointer = x.data();
    cxxblas::trmv(n, a_pointer, x_pointer, a_step, x_step,
                  cxxblas::Order::kRowMajor, uplo, cxxblas::Trans::kNoTrans,
                  diag);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             x.partialContiguity(0, x.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::trmv(n, &a_pointer[i * a_batch], &x_pointer[i * x_batch],
                    a_step, x_step, cxxblas::Order::kRowMajor, uplo,
                    cxxblas::Trans::kNoTrans, diag);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::trmv(n, a[x_begin.position()].data(), &(*x_begin),
                    a_step, x_step, cxxblas::Order::kRowMajor, uplo,
                    cxxblas::Trans::kNoTrans, diag);
    }
  }

  return x;
}

template < typename L >
const typename L::tensor_type& trsv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::Uplo uplo, typename L::Diag diag) {
  typedef typename L::tensor_type T;
  if (a.dimension() != x.dimension() + 1) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    if (a.size(i) != x.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.size(a.dimension() - 1) != x.size(x.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.size(a.dimension() - 2) != x.size(x.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous");
  }

  int n = static_cast< int >(a.size(a.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer x_pointer = x.data();
    cxxblas::trsv(n, a_pointer, x_pointer, a_step, x_step,
                  cxxblas::Order::kRowMajor, uplo, cxxblas::Trans::kNoTrans,
                  diag);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             x.partialContiguity(0, x.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::trsv(n, &a_pointer[i * a_batch], &x_pointer[i * x_batch],
                    a_step, x_step, cxxblas::Order::kRowMajor, uplo,
                    cxxblas::Trans::kNoTrans, diag);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::trsv(n, a[x_begin.position()].data(), &(*x_begin),
                    a_step, x_step, cxxblas::Order::kRowMajor, uplo,
                    cxxblas::Trans::kNoTrans, diag);
    }
  }

  return x;
}

template < typename L >
typename L::tensor_type* gbmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::tensor_type *y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::size_type kl,
    typename L::size_type ku) {
  typedef typename L::tensor_type T;
  if (a.dimension() >= 2) {
    typename T::size_storage y_size(a.dimension() - 1);
    for (typename T::dim_type i = 0; i < y_size.size(); ++i) {
      y_size[i] = a.size(i);
    }
    y->resize(y_size);
  }
  gbmv(l, a, x, y->zero(), alpha, beta, kl, ku);
  return y;
}

template < typename L >
typename L::tensor_type* gemv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::tensor_type *y, const typename L::value_type &alpha,
    const typename L::value_type &beta) {
  typedef typename L::tensor_type T;
  if (a.dimension() >= 2) {
    typename T::size_storage y_size(a.dimension() - 1);
    for (typename T::dim_type i = 0; i < y_size.size(); ++i) {
      y_size[i] = a.size(i);
    }
    y->resize(y_size);
  }
  gemv(l, a, x, y->zero(), alpha, beta);
  return y;
}

template < typename L >
typename L::tensor_type* ger(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    typename L::tensor_type *a, const typename L::value_type &alpha) {
  typedef typename L::tensor_type T;
  typename T::size_storage a_size(x.dimension() + 1);
  for (typename T::dim_type i = 0; i < x.dimension(); ++i) {
    a_size[i] = x.size(i);
  }
  a_size[a_size.size() - 1] = y.size(y.dimension() - 1);
  ger(l, x, y, a->resize(a_size).zero(), alpha);
  return a;
}

template < typename L >
typename L::tensor_type* gerc(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    typename L::tensor_type *a, const typename L::value_type &alpha) {
  typedef typename L::tensor_type T;
  typename T::size_storage a_size(x.dimension() + 1);
  for (typename T::dim_type i = 0; i < x.dimension(); ++i) {
    a_size[i] = x.size(i);
  }
  a_size[a_size.size() - 1] = y.size(y.dimension() - 1);
  gerc(l, x, y, a->resize(a_size).zero(), alpha);
  return a;
}

template < typename L >
typename L::tensor_type* hbmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::tensor_type *y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::size_type k,
    typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (a.dimension() >= 2) {
    typename T::size_storage y_size(a.dimension() - 1);
    for (typename T::dim_type i = 0; i < y_size.size(); ++i) {
      y_size[i] = a.size(i);
    }
    y->resize(y_size);
  }
  hbmv(l, a, x, y->zero(), alpha, beta, k, uplo);
  return y;
}

template < typename L >
typename L::tensor_type* hemv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::tensor_type *y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (a.dimension() >= 2) {
    typename T::size_storage y_size(a.dimension() - 1);
    for (typename T::dim_type i = 0; i < y_size.size(); ++i) {
      y_size[i] = a.size(i);
    }
    y->resize(y_size);
  }
  hemv(l, a, x, y->zero(), alpha, beta, uplo);
  return y;
}

template < typename L >
typename L::tensor_type* her(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *a,
    const typename L::real_type &alpha, typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  typename T::size_storage a_size(x.dimension() + 1);
  for (typename T::dim_type i = 0; i < x.dimension(); ++i) {
    a_size[i] = x.size(i);
  }
  a_size[a_size.size() - 1] = x.size(x.dimension() - 1);
  her(l, x, a->resize(a_size).zero(), alpha, uplo);
  return a;
}

template < typename L >
typename L::tensor_type* her2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    typename L::tensor_type *a, const typename L::value_type &alpha,
    typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  typename T::size_storage a_size(x.dimension() + 1);
  for (typename T::dim_type i = 0; i < x.dimension(); ++i) {
    a_size[i] = x.size(i);
  }
  a_size[a_size.size() - 1] = y.size(y.dimension() - 1);
  her2(l, x, y, a->resize(a_size).zero(), alpha, uplo);
  return a;
}

template < typename L >
typename L::tensor_type* hpmv(
    L *l, const typename L::tensor_type &ap, const typename L::tensor_type &x,
    typename L::tensor_type *y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo) {
  hpmv(l, ap, x, y->resizeAs(x).zero(), alpha, beta, uplo);
  return y;
}

template < typename L >
typename L::tensor_type* hpr(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *ap,
    const typename L::real_type &alpha, typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  typename T::size_storage ap_size(x.dimension());
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    ap_size[i] = x.size(i);
  }
  ap_size[ap_size.size() - 1] =
      (x.size(x.dimension() - 1) + 1) * x.size(x.dimension() - 1) / 2;
  hpr(l, x, ap->resize(ap_size), alpha, uplo);
  return ap;
}

template < typename L >
typename L::tensor_type* hpr2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    typename L::tensor_type *ap, const typename L::value_type &alpha,
    typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  typename T::size_storage ap_size(x.dimension());
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    ap_size[i] = x.size(i);
  }
  ap_size[ap_size.size() - 1] =
      (x.size(x.dimension() - 1) + 1) * x.size(x.dimension() - 1) / 2;
  hpr2(l, x, y, ap->resize(ap_size), alpha, uplo);
  return ap;
}

template < typename L >
typename L::tensor_type* sbmv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::tensor_type *y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::size_type k,
    typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (a.dimension() >= 2) {
    typename T::size_storage y_size(a.dimension() - 1);
    for (typename T::dim_type i = 0; i < y_size.size(); ++i) {
      y_size[i] = a.size(i);
    }
    y->resize(y_size);
  }
  sbmv(l, a, x, y->zero(), alpha, beta, k, uplo);
  return y;
}

template < typename L >
typename L::tensor_type* spmv(
    L *l, const typename L::tensor_type &ap, const typename L::tensor_type &x,
    typename L::tensor_type *y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo) {
  spmv(l, ap, x, y->resizeAs(x).zero(), alpha, beta, uplo);
  return y;
}

template < typename L >
typename L::tensor_type* spr(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *ap,
    const typename L::real_type &alpha, typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  typename T::size_storage ap_size(x.dimension());
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    ap_size[i] = x.size(i);
  }
  ap_size[ap_size.size() - 1] =
      (x.size(x.dimension() - 1) + 1) * x.size(x.dimension() - 1) / 2;
  spr(l, x, ap->resize(ap_size), alpha, uplo);
  return ap;
}

template < typename L >
typename L::tensor_type* spr2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    typename L::tensor_type *ap, const typename L::value_type &alpha,
    typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  typename T::size_storage ap_size(x.dimension());
  for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
    ap_size[i] = x.size(i);
  }
  ap_size[ap_size.size() - 1] =
      (x.size(x.dimension() - 1) + 1) * x.size(x.dimension() - 1) / 2;
  spr2(l, x, y, ap->resize(ap_size), alpha, uplo);
  return ap;
}

template < typename L >
typename L::tensor_type* symv(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &x,
    typename L::tensor_type *y, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (a.dimension() >= 2) {
    typename T::size_storage y_size(a.dimension() - 1);
    for (typename T::dim_type i = 0; i < y_size.size(); ++i) {
      y_size[i] = a.size(i);
    }
    y->resize(y_size);
  }
  symv(l, a, x, y->zero(), alpha, beta, uplo);
  return y;
}

template < typename L >
typename L::tensor_type* syr(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *a,
    const typename L::real_type &alpha, typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  typename T::size_storage a_size(x.dimension() + 1);
  for (typename T::dim_type i = 0; i < x.dimension(); ++i) {
    a_size[i] = x.size(i);
  }
  a_size[a_size.size() - 1] = x.size(x.dimension() - 1);
  syr(l, x, a->resize(a_size).zero(), alpha, uplo);
  return a;
}

template < typename L >
typename L::tensor_type* syr2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    typename L::tensor_type *a, const typename L::value_type &alpha,
    typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  typename T::size_storage a_size(x.dimension() + 1);
  for (typename T::dim_type i = 0; i < x.dimension(); ++i) {
    a_size[i] = x.size(i);
  }
  a_size[a_size.size() - 1] = y.size(y.dimension() - 1);
  syr2(l, x, y, a->resize(a_size).zero(), alpha, uplo);
  return a;
}

template < typename L >
typename L::tensor_type* tbmv(
    L *l, const typename L::tensor_type &a, typename L::tensor_type *x,
    typename L::size_type k, typename L::Uplo uplo,
    typename L::Diag diag) {
  typedef typename L::tensor_type T;
  if (a.dimension() >= 2) {
    typename T::size_storage x_size(a.dimension() - 1);
    for (typename T::dim_type i = 0; i < x_size.size(); ++i) {
      x_size[i] = a.size(i);
    }
    x->resize(x_size);
  }
  tbmv(l, a, x->zero(), k, uplo, diag);
  return x;
}

template < typename L >
typename L::tensor_type* tbsv(
    L *l, const typename L::tensor_type &a, typename L::tensor_type *x,
    typename L::size_type k, typename L::Uplo uplo,
    typename L::Diag diag) {
  typedef typename L::tensor_type T;
  if (a.dimension() >= 2) {
    typename T::size_storage x_size(a.dimension() - 1);
    for (typename T::dim_type i = 0; i < x_size.size(); ++i) {
      x_size[i] = a.size(i);
    }
    x->resize(x_size);
  }
  tbsv(l, a, x->zero(), k, uplo, diag);
  return x;
}

template < typename L >
typename L::tensor_type* tpmv(
    L *l, const typename L::tensor_type &ap, typename L::tensor_type *x,
    typename L::Uplo uplo, typename L::Diag diag) {
  typedef typename L::tensor_type T;
  typename T::size_storage x_size(ap.dimension());
  for (typename T::dim_type i = 0; i < ap.dimension() - 1; ++i) {
    x_size[i] = ap.size(i);
  }
  x_size[x_size.size() - 1] = (
      static_cast< typename T::size_type >(::std::sqrt(static_cast< double >(
          1 + 8 * ap.size(ap.dimension() - 1))))- 1) / 2;
  tpmv(l, ap, x->resize(x_size).zero(), uplo, diag);
  return x;
}

template < typename L >
typename L::tensor_type* tpsv(
    L *l, const typename L::tensor_type &ap, typename L::tensor_type *x,
    typename L::Uplo uplo, typename L::Diag diag) {
  typedef typename L::tensor_type T;
  typename T::size_storage x_size(ap.dimension());
  for (typename T::dim_type i = 0; i < ap.dimension() - 1; ++i) {
    x_size[i] = ap.size(i);
  }
  x_size[x_size.size() - 1] = (
      static_cast< typename T::size_type >(::std::sqrt(static_cast< double >(
          1 + 8 * ap.size(ap.dimension() - 1)))) - 1) / 2;
  tpsv(l, ap, x->resize(x_size).zero(), uplo, diag);
  return x;
}

template < typename L >
typename L::tensor_type* trmv(
    L *l, const typename L::tensor_type &a, typename L::tensor_type *x,
    typename L::Uplo uplo, typename L::Diag diag) {
  typedef typename L::tensor_type T;
  if (a.dimension() >= 2) {
    typename T::size_storage x_size(a.dimension() - 1);
    for (typename T::dim_type i = 0; i < x_size.size(); ++i) {
      x_size[i] = a.size(i);
    }
    x->resize(x_size);
  }
  trmv(l, a, x->zero(), uplo, diag);
  return x;
}

template < typename L >
typename L::tensor_type* trsv(
    L *l, const typename L::tensor_type &a, typename L::tensor_type *x,
    typename L::Uplo uplo, typename L::Diag diag) {
  typedef typename L::tensor_type T;
  if (a.dimension() >= 2) {
    typename T::size_storage x_size(a.dimension() - 1);
    for (typename T::dim_type i = 0; i < x_size.size(); ++i) {
      x_size[i] = a.size(i);
    }
    x->resize(x_size);
  }
  trsv(l, a, x->zero(), uplo, diag);
  return x;
}

}  // namespace math
}  // namespace linalg
}  // namespace thunder

#endif  // THUNDER_LINALG_MATH_INL_BLASLEVEL2_HPP_
