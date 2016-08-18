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

#ifndef THUNDER_LINALG_MATH_INL_BLASLEVEL3_HPP_
#define THUNDER_LINALG_MATH_INL_BLASLEVEL3_HPP_

#include "thunder/linalg/math.hpp"
#include "thunder/linalg/math-inl.hpp"

#include "thunder/exception.hpp"
#include "thunder/linalg/cxxblas.hpp"
#include "thunder/tensor.hpp"

namespace thunder {
namespace linalg {
namespace math {

template < typename L >
const typename L::tensor_type& gemm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type&b,
    const typename L::tensor_type &c, const typename L::value_type &alpha,
    const typename L::value_type &beta) {
  typedef typename L::tensor_type T;
  if (a.dimension() != b.dimension() || a.dimension() != c.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
    if (a.size(i) != b.size(i) || a.size(i) != c.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.size(a.dimension() - 2) != c.size(c.dimension() - 2) ||
      a.size(a.dimension() - 1) != b.size(b.dimension() - 2) ||
      b.size(b.dimension() - 1) != c.size(c.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1 || b.stride(b.dimension() - 1) != 1 ||
      c.stride(c.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous.");
  }

  int m = static_cast< int >(c.size(c.dimension() - 2));
  int n = static_cast< int >(c.size(c.dimension() - 1));
  int k = static_cast< int >(a.size(a.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int b_step = static_cast< int >(b.stride(b.dimension() - 2));
  int c_step = static_cast< int >(c.stride(c.dimension() - 2));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer b_pointer = b.data();
    typename T::pointer c_pointer = c.data();
    cxxblas::gemm(m, n, k, a_pointer, b_pointer, c_pointer, alpha, beta, a_step,
                  b_step, c_step);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             b.partialContiguity(0, b.dimension() - 3) &&
             c.partialContiguity(0, c.dimension() - 3)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer b_pointer = b.data();
    typename T::difference_type b_batch = b.stride(b.dimension() - 3);
    typename T::pointer c_pointer = c.data();
    typename T::difference_type c_batch = c.stride(c.dimension() - 3);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::gemm(m, n, k, &a_pointer[i * a_batch], &b_pointer[i * b_batch],
                    &c_pointer[i * c_batch], alpha, beta, a_step, b_step,
                    c_step);
    }
  } else {
    T a_select = a.select(a.dimension() - 1, 0).select(a.dimension() - 2, 0);
    for (typename T::reference_iterator a_begin = a_select.reference_begin(),
             a_end = a_select.reference_end(); a_begin != a_end; ++a_begin) {
      cxxblas::gemm(m, n, k, &(*a_begin), b[a_begin.position()].data(),
                    c[a_begin.position()].data(), alpha, beta, a_step, b_step,
                    c_step);
    }
  }
  return c;
}

template < typename L >
const typename L::tensor_type& hemm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type&b,
    const typename L::tensor_type &c, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Side side,
    typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (a.dimension() != b.dimension() || a.dimension() != c.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
    if (a.size(i) != b.size(i) || a.size(i) != c.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (c.size(c.dimension() - 1) != c.size(c.dimension() - 2)) {
    throw out_of_range("Matrix size is not symmetric.");
  }
  if (side == L::Side::kLeft) {
    if (a.size(a.dimension() - 2) != c.size(c.dimension() - 2) ||
        a.size(a.dimension() - 1) != b.size(b.dimension() - 2) ||
        b.size(b.dimension() - 1) != c.size(c.dimension() - 1)) {
      throw out_of_range("Tensor size does not match.");
    }
  } else {
    if (a.size(a.dimension() - 2) != b.size(b.dimension() - 1) ||
        a.size(a.dimension() - 1) != c.size(b.dimension() - 1) ||
        b.size(b.dimension() - 2) != c.size(c.dimension() - 2)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.stride(a.dimension() - 1) != 1 || b.stride(b.dimension() - 1) != 1 ||
      c.stride(c.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous.");
  }

  int m = static_cast< int >(c.size(c.dimension() - 2));
  int n = static_cast< int >(c.size(c.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int b_step = static_cast< int >(b.stride(b.dimension() - 2));
  int c_step = static_cast< int >(c.stride(c.dimension() - 2));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer b_pointer = b.data();
    typename T::pointer c_pointer = c.data();
    cxxblas::hemm(m, n, a_pointer, b_pointer, c_pointer, alpha, beta, a_step,
                  b_step, c_step, cxxblas::Order::kRowMajor, side, uplo);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             b.partialContiguity(0, b.dimension() - 3) &&
             c.partialContiguity(0, c.dimension() - 3)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer b_pointer = b.data();
    typename T::difference_type b_batch = b.stride(b.dimension() - 3);
    typename T::pointer c_pointer = c.data();
    typename T::difference_type c_batch = c.stride(c.dimension() - 3);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::hemm(m, n, &a_pointer[i * a_batch], &b_pointer[i * b_batch],
                    &c_pointer[i * c_batch], alpha, beta, a_step, b_step,
                    c_step, cxxblas::Order::kRowMajor, side, uplo);
    }
  } else {
    T a_select = a.select(a.dimension() - 1, 0).select(a.dimension() - 2, 0);
    for (typename T::reference_iterator a_begin = a_select.reference_begin(),
             a_end = a_select.reference_end(); a_begin != a_end; ++a_begin) {
      cxxblas::hemm(m, n, &(*a_begin), b[a_begin.position()].data(),
                    c[a_begin.position()].data(), alpha, beta, a_step, b_step,
                    c_step, cxxblas::Order::kRowMajor, side, uplo);
    }
  }

  return c;
}

template < typename L >
const typename L::tensor_type& herk(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &c,
    const typename L::real_type &alpha, const typename L::real_type &beta,
    typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (a.dimension() != c.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
    if (a.size(i) != c.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (c.size(c.dimension() - 1) != c.size(c.dimension() - 2)) {
    throw out_of_range("Matrix size is not symmetric.");
  }
  if (a.size(a.dimension() - 2) != c.size(c.dimension() - 2)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1 || c.stride(c.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous.");
  }

  int n = static_cast< int >(c.size(c.dimension() - 1));
  int k = static_cast< int >(a.size(a.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int c_step = static_cast< int >(c.stride(c.dimension() - 2));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer c_pointer = c.data();
    cxxblas::herk(n, k, a_pointer, c_pointer, alpha, beta, a_step, c_step,
                  cxxblas::Order::kRowMajor, uplo);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             c.partialContiguity(0, c.dimension() - 3)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer c_pointer = c.data();
    typename T::difference_type c_batch = c.stride(c.dimension() - 3);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::herk(n, k, &a_pointer[i * a_batch], &c_pointer[i * c_batch],
                    alpha, beta, a_step, c_step, cxxblas::Order::kRowMajor,
                    uplo);
    }
  } else {
    T a_select = a.select(a.dimension() - 1, 0).select(a.dimension() - 2, 0);
    for (typename T::reference_iterator a_begin = a_select.reference_begin(),
             a_end = a_select.reference_end(); a_begin != a_end; ++a_begin) {
      cxxblas::herk(n, k, &(*a_begin), c[a_begin.position()].data(), alpha,
                    beta, a_step, c_step, cxxblas::Order::kRowMajor, uplo);
    }
  }

  return c;
}

template < typename L >
const typename L::tensor_type& her2k(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type&b,
    const typename L::tensor_type &c, const typename L::value_type &alpha,
    const typename L::real_type &beta, typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (a.dimension() != b.dimension() || a.dimension() != c.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
    if (a.size(i) != b.size(i) || a.size(i) != c.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (c.size(c.dimension() - 1) != c.size(c.dimension() - 2)) {
    throw out_of_range("Matrix size is not symmetric.");
  }
  if (a.size(a.dimension() - 2) != c.size(c.dimension() - 2) ||
      a.size(a.dimension() - 1) != b.size(b.dimension() - 1) ||
      b.size(b.dimension() - 2) != c.size(c.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1 || b.stride(b.dimension() - 1) != 1 ||
      c.stride(c.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous.");
  }

  int n = static_cast< int >(c.size(c.dimension() - 2));
  int k = static_cast< int >(a.size(a.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int b_step = static_cast< int >(b.stride(b.dimension() - 2));
  int c_step = static_cast< int >(c.stride(c.dimension() - 2));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer b_pointer = b.data();
    typename T::pointer c_pointer = c.data();
    cxxblas::her2k(n, k, a_pointer, b_pointer, c_pointer, alpha, beta, a_step,
                   b_step, c_step, cxxblas::Order::kRowMajor, uplo);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             b.partialContiguity(0, b.dimension() - 3) &&
             c.partialContiguity(0, c.dimension() - 3)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer b_pointer = b.data();
    typename T::difference_type b_batch = b.stride(b.dimension() - 3);
    typename T::pointer c_pointer = c.data();
    typename T::difference_type c_batch = c.stride(c.dimension() - 3);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::her2k(n, k, &a_pointer[i * a_batch], &b_pointer[i * b_batch],
                     &c_pointer[i * c_batch], alpha, beta, a_step, b_step,
                     c_step, cxxblas::Order::kRowMajor, uplo);
    }
  } else {
    T a_select = a.select(a.dimension() - 1, 0).select(a.dimension() - 2, 0);
    for (typename T::reference_iterator a_begin = a_select.reference_begin(),
             a_end = a_select.reference_end(); a_begin != a_end; ++a_begin) {
      cxxblas::her2k(n, k, &(*a_begin), b[a_begin.position()].data(),
                     c[a_begin.position()].data(), alpha, beta, a_step, b_step,
                     c_step, cxxblas::Order::kRowMajor, uplo);
    }
  }

  return c;
}

template < typename L >
const typename L::tensor_type& symm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type&b,
    const typename L::tensor_type &c, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Side side,
    typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (a.dimension() != b.dimension() || a.dimension() != c.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
    if (a.size(i) != b.size(i) || a.size(i) != c.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (c.size(c.dimension() - 1) != c.size(c.dimension() - 2)) {
    throw out_of_range("Matrix size is not symmetric.");
  }
  if (side == L::Side::kLeft) {
    if (a.size(a.dimension() - 2) != c.size(c.dimension() - 2) ||
        a.size(a.dimension() - 1) != b.size(b.dimension() - 2) ||
        b.size(b.dimension() - 1) != c.size(c.dimension() - 1)) {
      throw out_of_range("Tensor size does not match.");
    }
  } else {
    if (a.size(a.dimension() - 2) != b.size(b.dimension() - 1) ||
        a.size(a.dimension() - 1) != c.size(b.dimension() - 1) ||
        b.size(b.dimension() - 2) != c.size(c.dimension() - 2)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.stride(a.dimension() - 1) != 1 || b.stride(b.dimension() - 1) != 1 ||
      c.stride(c.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous.");
  }

  int m = static_cast< int >(c.size(c.dimension() - 2));
  int n = static_cast< int >(c.size(c.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int b_step = static_cast< int >(b.stride(b.dimension() - 2));
  int c_step = static_cast< int >(c.stride(c.dimension() - 2));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer b_pointer = b.data();
    typename T::pointer c_pointer = c.data();
    cxxblas::symm(m, n, a_pointer, b_pointer, c_pointer, alpha, beta, a_step,
                  b_step, c_step, cxxblas::Order::kRowMajor, side, uplo);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             b.partialContiguity(0, b.dimension() - 3) &&
             c.partialContiguity(0, c.dimension() - 3)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer b_pointer = b.data();
    typename T::difference_type b_batch = b.stride(b.dimension() - 3);
    typename T::pointer c_pointer = c.data();
    typename T::difference_type c_batch = c.stride(c.dimension() - 3);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::symm(m, n, &a_pointer[i * a_batch], &b_pointer[i * b_batch],
                    &c_pointer[i * c_batch], alpha, beta, a_step, b_step,
                    c_step, cxxblas::Order::kRowMajor, side, uplo);
    }
  } else {
    T a_select = a.select(a.dimension() - 1, 0).select(a.dimension() - 2, 0);
    for (typename T::reference_iterator a_begin = a_select.reference_begin(),
             a_end = a_select.reference_end(); a_begin != a_end; ++a_begin) {
      cxxblas::symm(m, n, &(*a_begin), b[a_begin.position()].data(),
                    c[a_begin.position()].data(), alpha, beta, a_step, b_step,
                    c_step, cxxblas::Order::kRowMajor, side, uplo);
    }
  }

  return c;
}

template < typename L >
const typename L::tensor_type& syrk(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &c,
    const typename L::value_type &alpha, const typename L::value_type &beta,
    typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (a.dimension() != c.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
    if (a.size(i) != c.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (c.size(c.dimension() - 1) != c.size(c.dimension() - 2)) {
    throw out_of_range("Matrix size is not symmetric.");
  }
  if (a.size(a.dimension() - 2) != c.size(c.dimension() - 2)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1 || c.stride(c.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous.");
  }

  int n = static_cast< int >(c.size(c.dimension() - 1));
  int k = static_cast< int >(a.size(a.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int c_step = static_cast< int >(c.stride(c.dimension() - 2));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer c_pointer = c.data();
    cxxblas::syrk(n, k, a_pointer, c_pointer, alpha, beta, a_step, c_step,
                  cxxblas::Order::kRowMajor, uplo);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             c.partialContiguity(0, c.dimension() - 3)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer c_pointer = c.data();
    typename T::difference_type c_batch = c.stride(c.dimension() - 3);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::syrk(n, k, &a_pointer[i * a_batch], &c_pointer[i * c_batch],
                    alpha, beta, a_step, c_step, cxxblas::Order::kRowMajor,
                    uplo);
    }
  } else {
    T a_select = a.select(a.dimension() - 1, 0).select(a.dimension() - 2, 0);
    for (typename T::reference_iterator a_begin = a_select.reference_begin(),
             a_end = a_select.reference_end(); a_begin != a_end; ++a_begin) {
      cxxblas::syrk(n, k, &(*a_begin), c[a_begin.position()].data(), alpha,
                    beta, a_step, c_step, cxxblas::Order::kRowMajor, uplo);
    }
  }

  return c;
}

template < typename L >
const typename L::tensor_type& syr2k(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type&b,
    const typename L::tensor_type &c, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  if (a.dimension() != b.dimension() || a.dimension() != c.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
    if (a.size(i) != b.size(i) || a.size(i) != c.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (c.size(c.dimension() - 1) != c.size(c.dimension() - 2)) {
    throw out_of_range("Matrix size is not symmetric.");
  }
  if (a.size(a.dimension() - 2) != c.size(c.dimension() - 2) ||
      a.size(a.dimension() - 1) != b.size(b.dimension() - 1) ||
      b.size(b.dimension() - 2) != c.size(c.dimension() - 1)) {
    throw out_of_range("Tensor size does not match.");
  }
  if (a.stride(a.dimension() - 1) != 1 || b.stride(b.dimension() - 1) != 1 ||
      c.stride(c.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous.");
  }

  int n = static_cast< int >(c.size(c.dimension() - 2));
  int k = static_cast< int >(a.size(a.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int b_step = static_cast< int >(b.stride(b.dimension() - 2));
  int c_step = static_cast< int >(c.stride(c.dimension() - 2));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer b_pointer = b.data();
    typename T::pointer c_pointer = c.data();
    cxxblas::syr2k(n, k, a_pointer, b_pointer, c_pointer, alpha, beta, a_step,
                   b_step, c_step, cxxblas::Order::kRowMajor, uplo);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             b.partialContiguity(0, b.dimension() - 3) &&
             c.partialContiguity(0, c.dimension() - 3)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer b_pointer = b.data();
    typename T::difference_type b_batch = b.stride(b.dimension() - 3);
    typename T::pointer c_pointer = c.data();
    typename T::difference_type c_batch = c.stride(c.dimension() - 3);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::syr2k(n, k, &a_pointer[i * a_batch], &b_pointer[i * b_batch],
                     &c_pointer[i * c_batch], alpha, beta, a_step, b_step,
                     c_step, cxxblas::Order::kRowMajor, uplo);
    }
  } else {
    T a_select = a.select(a.dimension() - 1, 0).select(a.dimension() - 2, 0);
    for (typename T::reference_iterator a_begin = a_select.reference_begin(),
             a_end = a_select.reference_end(); a_begin != a_end; ++a_begin) {
      cxxblas::syr2k(n, k, &(*a_begin), b[a_begin.position()].data(),
                     c[a_begin.position()].data(), alpha, beta, a_step, b_step,
                     c_step, cxxblas::Order::kRowMajor, uplo);
    }
  }

  return c;
}

template < typename L >
const typename L::tensor_type& trmm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type&b,
    const typename L::value_type &alpha, typename L::Side side,
    typename L::Uplo uplo, typename L::Diag diag) {
  typedef typename L::tensor_type T;
  if (a.dimension() != b.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
    if (a.size(i) != b.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (side == L::Side::kLeft) {
    if (a.size(a.dimension() - 2) != a.size(a.dimension() - 1) ||
        a.size(a.dimension() - 1) != b.size(b.dimension() - 2)) {
      throw out_of_range("Tensor size does not match.");
    }
  } else {
    if (a.size(a.dimension() - 2) != a.size(a.dimension() - 1) ||
        a.size(a.dimension() - 2) != b.size(b.dimension() - 1)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.stride(a.dimension() - 1) != 1 || b.stride(b.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous.");
  }

  int m = static_cast< int >(b.size(b.dimension() - 2));
  int n = static_cast< int >(b.size(b.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int b_step = static_cast< int >(b.stride(b.dimension() - 2));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer b_pointer = b.data();
    cxxblas::trmm(m, n, a_pointer, b_pointer, alpha, a_step, b_step,
                  cxxblas::Order::kRowMajor, side, uplo,
                  cxxblas::Trans::kNoTrans, diag);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             b.partialContiguity(0, b.dimension() - 3)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer b_pointer = b.data();
    typename T::difference_type b_batch = b.stride(b.dimension() - 3);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::trmm(m, n, &a_pointer[i * a_batch], &b_pointer[i * b_batch],
                    alpha, a_step, b_step, cxxblas::Order::kRowMajor, side,
                    uplo, cxxblas::Trans::kNoTrans, diag);
    }
  } else {
    T a_select = a.select(a.dimension() - 1, 0).select(a.dimension() - 2, 0);
    for (typename T::reference_iterator a_begin = a_select.reference_begin(),
             a_end = a_select.reference_end(); a_begin != a_end; ++a_begin) {
      cxxblas::trmm(m, n, &(*a_begin), b[a_begin.position()].data(), alpha,
                    a_step, b_step, cxxblas::Order::kRowMajor, side, uplo,
                    cxxblas::Trans::kNoTrans, diag);
    }
  }
  return b;
}

template < typename L >
const typename L::tensor_type& trsm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &b,
    const typename L::value_type &alpha, typename L::Side side,
    typename L::Uplo uplo, typename L::Diag diag) {
  typedef typename L::tensor_type T;
  if (a.dimension() != b.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
    if (a.size(i) != b.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (side == L::Side::kLeft) {
    if (a.size(a.dimension() - 2) != a.size(a.dimension() - 1) ||
        a.size(a.dimension() - 1) != b.size(b.dimension() - 2)) {
      throw out_of_range("Tensor size does not match.");
    }
  } else {
    if (a.size(a.dimension() - 2) != a.size(a.dimension() - 1) ||
        a.size(a.dimension() - 2) != b.size(b.dimension() - 1)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (a.stride(a.dimension() - 1) != 1 || b.stride(b.dimension() - 1) != 1) {
    throw contiguity_error("Last dimension of matrix is not contiguous.");
  }

  int m = static_cast< int >(b.size(b.dimension() - 2));
  int n = static_cast< int >(b.size(b.dimension() - 1));
  int a_step = static_cast< int >(a.stride(a.dimension() - 2));
  int b_step = static_cast< int >(b.stride(b.dimension() - 2));

  if (a.dimension() == 2) {
    typename T::pointer a_pointer = a.data();
    typename T::pointer b_pointer = b.data();
    cxxblas::trsm(m, n, a_pointer, b_pointer, alpha, a_step, b_step,
                  cxxblas::Order::kRowMajor, side, uplo,
                  cxxblas::Trans::kNoTrans, diag);
  } else if (a.partialContiguity(0, a.dimension() - 3) &&
             b.partialContiguity(0, b.dimension() - 3)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < a.dimension() - 2; ++i) {
      batch_size = batch_size * a.size(i);
    }
    typename T::pointer a_pointer = a.data();
    typename T::difference_type a_batch = a.stride(a.dimension() - 3);
    typename T::pointer b_pointer = b.data();
    typename T::difference_type b_batch = b.stride(b.dimension() - 3);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::trsm(m, n, &a_pointer[i * a_batch], &b_pointer[i * b_batch],
                    alpha, a_step, b_step, cxxblas::Order::kRowMajor, side,
                    uplo, cxxblas::Trans::kNoTrans, diag);
    }
  } else {
    T a_select = a.select(a.dimension() - 1, 0).select(a.dimension() - 2, 0);
    for (typename T::reference_iterator a_begin = a_select.reference_begin(),
             a_end = a_select.reference_end(); a_begin != a_end; ++a_begin) {
      cxxblas::trsm(m, n, &(*a_begin), b[a_begin.position()].data(), alpha,
                    a_step, b_step, cxxblas::Order::kRowMajor, side, uplo,
                    cxxblas::Trans::kNoTrans, diag);
    }
  }
  return b;
}

template < typename L >
typename L::tensor_type* gemm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &b,
    typename L::tensor_type *c, const typename L::value_type &alpha,
    const typename L::value_type &beta) {
  typedef typename L::tensor_type T;
  typename T::size_storage c_size(a.dimension());
  if (a.dimension() >= 2) {
    for (typename T::dim_type i = 0; i < c_size.size() - 1; ++i) {
      c_size[i] = a.size(i);
    }
    c_size[c_size.size() - 1] = b.size(b.dimension() - 1);
  }
  gemm(l, a, b, c->resize(c_size).zero(), alpha, beta);
  return c;
}

template < typename L >
typename L::tensor_type* hemm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &b,
    typename L::tensor_type *c, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Side side,
    typename L::Uplo uplo) {
  hemm(l, a, b, c->resizeAs(b).zero(), alpha, beta, side, uplo);
  return c;
}

template < typename L >
typename L::tensor_type* herk(
    L *l, const typename L::tensor_type &a, typename L::tensor_type *c,
    const typename L::real_type &alpha, const typename L::real_type &beta,
    typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  typename T::size_storage c_size(a.dimension());
  if (a.dimension() >= 2) {
    for (typename T::dim_type i = 0; i < c_size.size() - 1; ++i) {
      c_size[i] = a.size(i);
    }
    c_size[c_size.size() - 1] = a.size(a.dimension() - 2);
  }
  herk(l, a, c->resize(c_size).zero(), alpha, beta, uplo);
  return c;
}

template < typename L >
typename L::tensor_type* her2k(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &b,
    typename L::tensor_type *c, const typename L::value_type &alpha,
    const typename L::real_type &beta, typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  typename T::size_storage c_size(a.dimension());
  if (a.dimension() >= 2) {
    for (typename T::dim_type i = 0; i < c_size.size() - 1; ++i) {
      c_size[i] = a.size(i);
    }
    c_size[c_size.size() - 1] = a.size(a.dimension() - 2);
  }
  her2k(l, a, b, c->resize(c_size).zero(), alpha, beta, uplo);
  return c;
}

template < typename L >
typename L::tensor_type* symm(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &b,
    typename L::tensor_type *c, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Side side,
    typename L::Uplo uplo) {
  symm(l, a, b, c->resizeAs(b).zero(), alpha, beta, side, uplo);
  return c;
}

template < typename L >
typename L::tensor_type* syrk(
    L *l, const typename L::tensor_type &a, typename L::tensor_type *c,
    const typename L::value_type &alpha, const typename L::value_type &beta,
    typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  typename T::size_storage c_size(a.dimension());
  if (a.dimension() >= 2) {
    for (typename T::dim_type i = 0; i < c_size.size() - 1; ++i) {
      c_size[i] = a.size(i);
    }
    c_size[c_size.size() - 1] = a.size(a.dimension() - 2);
  }
  syrk(l, a, c->resize(c_size).zero(), alpha, beta, uplo);
  return c;
}

template < typename L >
typename L::tensor_type* syr2k(
    L *l, const typename L::tensor_type &a, const typename L::tensor_type &b,
    typename L::tensor_type *c, const typename L::value_type &alpha,
    const typename L::value_type &beta, typename L::Uplo uplo) {
  typedef typename L::tensor_type T;
  typename T::size_storage c_size(a.dimension());
  if (a.dimension() >= 2) {
    for (typename T::dim_type i = 0; i < c_size.size() - 1; ++i) {
      c_size[i] = a.size(i);
    }
    c_size[c_size.size() - 1] = a.size(a.dimension() - 2);
  }
  syr2k(l, a, b, c->resize(c_size).zero(), alpha, beta, uplo);
  return c;
}

template < typename L >
typename L::tensor_type* trmm(
    L *l, const typename L::tensor_type &a, typename L::tensor_type *b,
    const typename L::value_type &alpha, typename L::Side side,
    typename L::Uplo uplo, typename L::Diag diag) {
  trmm(l, a, b->resizeAs(a).zero(), alpha, side, uplo, diag);
  return b;
}

template < typename L >
typename L::tensor_type* trsm(
    L *l, const typename L::tensor_type &a, typename L::tensor_type *b,
    const typename L::value_type &alpha, typename L::Side side,
    typename L::Uplo uplo, typename L::Diag diag) {
  trsm(l, a, b->resizeAs(a).zero(), alpha, side, uplo, diag);
  return b;
}

}  // namespace math
}  // namespace linalg
}  // namespace thunder

#endif  // THUNDER_LINALG_MATH_INL_BLASLEVEL3_HPP_
