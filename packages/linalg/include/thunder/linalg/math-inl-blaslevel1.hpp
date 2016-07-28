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

#ifndef THUNDER_LINALG_MATH_INL_BLASLEVEL1_HPP_
#define THUNDER_LINALG_MATH_INL_BLASLEVEL1_HPP_

#include "thunder/linalg/math.hpp"
#include "thunder/linalg/math-inl.hpp"

#include "thunder/exception.hpp"
#include "thunder/linalg/cxxblas.hpp"
#include "thunder/tensor.hpp"

namespace thunder {
namespace linalg {
namespace math {

template < typename L >
const typename L::tensor_type& asum(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &r) {
  typedef typename L::tensor_type T;
  if (x.dimension() == 1) {
    if (r.dimension() != 1 || r.size(0) != 1) {
      throw out_of_range("Result dimension does not match.");
    }
  } else if (r.dimension() != x.dimension() - 1) {
    throw out_of_range("Result dimension does not match.");
  } else {
    for (typename T::dim_type i = 0; i < r.dimension(); ++i) {
      if (r.size(i) != x.size(i)) {
        throw out_of_range("Result size does not match.");
      }
    }
  }

  int x_size = static_cast< int >(x.size(x.dimension() - 1));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  if (x.dimension() == 1) {
    typename T::pointer x_pointer = x.data();
    typename T::pointer r_pointer = r.data();
    r_pointer[0] = cxxblas::asum(x_size, x_pointer, x_step);
  } else if (x.partialContiguity(0, x.dimension() - 2) &&
             r.partialContiguity(0, r.dimension() - 1)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < r.dimension(); ++i) {
      batch_size = batch_size * r.size(i);
    }
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer r_pointer = r.data();
    typename T::difference_type r_batch = r.stride(r.dimension() - 1);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      r_pointer[i * r_batch] = cxxblas::asum(
          x_size, &x_pointer[i * x_batch], x_step);
    }
  } else {
    for (typename T::reference_iterator r_begin = r.reference_begin(),
             r_end = r.reference_end(); r_begin != r_end; ++r_begin) {
      *r_begin = cxxblas::asum(x_size, x[r_begin.position()].data(), x_step);
    }
  }
  return r;
}

template < typename L >
const typename L::tensor_type& axpy(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::value_type &a) {
  typedef typename L::tensor_type T;
  if (x.dimension() != y.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension(); ++i) {
    if (x.size(i) != y.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }

  int x_size = static_cast< int >(x.size(x.dimension() - 1));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));
  if (x.dimension() == 1) {
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    cxxblas::axpy(x_size, x_pointer, y_pointer, a, x_step, y_step);
  } else if (x.partialContiguity(0, x.dimension() - 2) &&
             y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
      batch_size = batch_size * x.size(i);
    }
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::axpy(x_size, &x_pointer[i * x_batch], &y_pointer[i * y_batch], a,
                    x_step, y_step);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::axpy(x_size, &(*x_begin), y[x_begin.position()].data(), a,
                    x_step, y_step);
    }
  }
  return y;
}

template < typename L >
const typename L::tensor_type& copy(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &r) {
  typedef typename L::tensor_type T;
  if (x.dimension() != r.dimension()) {
    throw out_of_range("Result dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension(); ++i) {
    if (x.size(i) != r.size(i)) {
      throw out_of_range("Result size does not match.");
    }
  }

  int x_size = static_cast< int >(x.size(x.dimension() - 1));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int r_step = static_cast< int >(r.stride(r.dimension() - 1));
  if (x.dimension() == 1) {
    typename T::pointer x_pointer = x.data();
    typename T::pointer r_pointer = r.data();
    cxxblas::copy(x_size, x_pointer, r_pointer, x_step, r_step);
  } else if (x.partialContiguity(0, x.dimension() - 2) &&
      r.partialContiguity(0, r.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
      batch_size = batch_size * x.size(i);
    }
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer r_pointer = r.data();
    typename T::difference_type r_batch = r.stride(r.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::copy(x_size, &x_pointer[i * x_batch], &r_pointer[i * r_batch],
                    x_step, r_step);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::copy(x_size, &(*x_begin), r[x_begin.position()].data(), x_step,
                    r_step);
    }
  }
  return r;
}

template < typename L >
const typename L::tensor_type& dot(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &r) {
  typedef typename L::tensor_type T;
  if (x.dimension() != y.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension(); ++i) {
    if (x.size(i) != y.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (x.dimension() == 1) {
    if (r.dimension() != 1 || r.size(0) != 1) {
      throw out_of_range("Result dimension does not match.");
    }
  } else if (r.dimension() != x.dimension() - 1) {
    throw out_of_range("Result dimension does not match.");
  } else {
    for (typename T::dim_type i = 0; i < r.dimension(); ++i) {
      if (r.size(i) != x.size(i)) {
        throw out_of_range("Result size does not match.");
      }
    }
  }

  int x_size = static_cast< int >(x.size(x.dimension() - 1));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));
  if (x.dimension() == 1) {
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    typename T::pointer r_pointer = r.data();
    r_pointer[0] = cxxblas::dot(x_size, x_pointer, y_pointer, x_step, y_step);
  } else if (x.partialContiguity(0, x.dimension() - 2) &&
      y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
      batch_size = batch_size * x.size(i);
    }
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    typename T::pointer r_pointer = r.data();
    typename T::difference_type r_batch = r.stride(r.dimension() - 1);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      r_pointer[i * r_batch] = cxxblas::dot(
          x_size, &x_pointer[i * x_batch], &y_pointer[i * y_batch], x_step,
          y_step);
    }
  } else {
    for (typename T::reference_iterator r_begin = r.reference_begin(),
             r_end = r.reference_end(); r_begin != r_end; ++r_begin) {
      *r_begin = cxxblas::dot(
          x_size, x[r_begin.position()].data(), y[r_begin.position()].data(),
          x_step, y_step);
    }
  }
  return r;
}

template < typename L >
const typename L::tensor_type& dotc(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &r) {
  typedef typename L::tensor_type T;
  if (x.dimension() != y.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension(); ++i) {
    if (x.size(i) != y.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (x.dimension() == 1) {
    if (r.dimension() != 1 || r.size(0) != 1) {
      throw out_of_range("Result dimension does not match.");
    }
  } else if (r.dimension() != x.dimension() - 1) {
    throw out_of_range("Result dimension does not match.");
  } else {
    for (typename T::dim_type i = 0; i < r.dimension(); ++i) {
      if (r.size(i) != x.size(i)) {
        throw out_of_range("Result size does not match.");
      }
    }
  }

  int x_size = static_cast< int >(x.size(x.dimension() - 1));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));
  if (x.dimension() == 1) {
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    typename T::pointer r_pointer = r.data();
    r_pointer[0] = cxxblas::dotc(x_size, x_pointer, y_pointer, x_step, y_step);
  } else if (x.partialContiguity(0, x.dimension() - 2) &&
      y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
      batch_size = batch_size * x.size(i);
    }
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    typename T::pointer r_pointer = r.data();
    typename T::difference_type r_batch = r.stride(r.dimension() - 1);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      r_pointer[i * r_batch] = cxxblas::dotc(
          x_size, &x_pointer[i * x_batch], &y_pointer[i * y_batch], x_step,
          y_step);
    }
  } else {
    for (typename T::reference_iterator r_begin = r.reference_begin(),
             r_end = r.reference_end(); r_begin != r_end; ++r_begin) {
      *r_begin = cxxblas::dotc(
          x_size, x[r_begin.position()].data(), y[r_begin.position()].data(),
          x_step, y_step);
    }
  }
  return r;
}

template < typename L >
const typename L::tensor_type& nrm2(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &r) {
  typedef typename L::tensor_type T;
  if (x.dimension() == 1) {
    if (r.dimension() != 1 || r.size(0) != 1) {
      throw out_of_range("Result dimension does not match.");
    }
  } else if (r.dimension() != x.dimension() - 1) {
    throw out_of_range("Result dimension does not match.");
  } else {
    for (typename T::dim_type i = 0; i < r.dimension(); ++i) {
      if (r.size(i) != x.size(i)) {
        throw out_of_range("Result size does not match.");
      }
    }
  }

  int x_size = static_cast< int >(x.size(x.dimension() - 1));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  if (x.dimension() == 1) {
    typename T::pointer x_pointer = x.data();
    typename T::pointer r_pointer = r.data();
    r_pointer[0] = cxxblas::nrm2(x_size, x_pointer, x_step);
  } else if (x.partialContiguity(0, x.dimension() - 2) &&
             r.partialContiguity(0, r.dimension() - 1)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < r.dimension(); ++i) {
      batch_size = batch_size * r.size(i);
    }
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer r_pointer = r.data();
    typename T::difference_type r_batch = r.stride(r.dimension() - 1);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      r_pointer[i * r_batch] = cxxblas::nrm2(
          x_size, &x_pointer[i * x_batch], x_step);
    }
  } else {
    for (typename T::reference_iterator r_begin = r.reference_begin(),
             r_end = r.reference_end(); r_begin != r_end; ++r_begin) {
      *r_begin = cxxblas::nrm2(x_size, x[r_begin.position()].data(), x_step);
    }
  }
  return r;
}

template < typename L >
const typename L::tensor_type& rot(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::value_type &c, const typename L::value_type &s) {
    typedef typename L::tensor_type T;
  if (x.dimension() != y.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension(); ++i) {
    if (x.size(i) != y.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }

  int x_size = static_cast< int >(x.size(x.dimension() - 1));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));
  if (x.dimension() == 1) {
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    cxxblas::rot(x_size, x_pointer, y_pointer, ::std::real(c), ::std::real(s),
                 x_step, y_step);
  } else if (x.partialContiguity(0, x.dimension() - 2) &&
      y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
      batch_size = batch_size * x.size(i);
    }
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::rot(x_size, &x_pointer[i * x_batch], &y_pointer[i * y_batch],
                   ::std::real(c), ::std::real(s), x_step, y_step);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::rot(x_size, &(*x_begin), y[x_begin.position()].data(),
                   ::std::real(c), ::std::real(s), x_step, y_step);
    }
  }
  return y;
}

template < typename L >
const typename L::tensor_type& rotm(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    const typename L::tensor_type &p) {
  typedef typename L::tensor_type T;
  if (x.dimension() != y.dimension()) {
    throw out_of_range("Tensor dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension(); ++i) {
    if (x.size(i) != y.size(i)) {
      throw out_of_range("Tensor size does not match.");
    }
  }
  if (p.dimension() != 1 || p.size(0) != 5) {
    throw out_of_range("Parameter size is not 5.");
  }

  int x_size = static_cast< int >(x.size(x.dimension() - 1));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));
  typename T::pointer p_pointer = p.data();
  if (x.dimension() == 1) {
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    cxxblas::rotm(x_size, x_pointer, y_pointer, p_pointer, x_step, y_step);
  } else if (x.partialContiguity(0, x.dimension() - 2) &&
      y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
      batch_size = batch_size * x.size(i);
    }
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::rotm(x_size, &x_pointer[i * x_batch], &y_pointer[i * y_batch],
                   p_pointer, x_step, y_step);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::rotm(x_size, &(*x_begin), y[x_begin.position()].data(),
                    p_pointer, x_step, y_step);
    }
  }
  return y;
}

template < typename L >
const typename L::tensor_type& scal(
    L *l, const typename L::tensor_type &x, const typename L::value_type &a) {
  typedef typename L::tensor_type T;
  int x_size = static_cast< int >(x.size(x.dimension() - 1));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  if (x.dimension() == 1) {
    typename T::pointer x_pointer = x.data();
    cxxblas::scal(x_size, x_pointer, a, x_step);
  } else if (x.partialContiguity(0, x.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
      batch_size = batch_size * x.size(i);
    }
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::scal(x_size, &x_pointer[i * x_batch], a, x_step);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::scal(x_size, &(*x_begin), a, x_step);
    }
  }
  return x;
}

template < typename L >
const typename L::tensor_type& swap(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y) {
  typedef typename L::tensor_type T;
  if (x.dimension() != y.dimension()) {
    throw out_of_range("Result dimension does not match.");
  }
  for (typename T::dim_type i = 0; i < x.dimension(); ++i) {
    if (x.size(i) != y.size(i)) {
      throw out_of_range("Result size does not match.");
    }
  }

  int x_size = static_cast< int >(x.size(x.dimension() - 1));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  int y_step = static_cast< int >(y.stride(y.dimension() - 1));
  if (x.dimension() == 1) {
    typename T::pointer x_pointer = x.data();
    typename T::pointer y_pointer = y.data();
    cxxblas::swap(x_size, x_pointer, y_pointer, x_step, y_step);
  } else if (x.partialContiguity(0, x.dimension() - 2) &&
      y.partialContiguity(0, y.dimension() - 2)) {
    typename T::size_type batch_size = 1;
    for (typename T::dim_type i = 0; i < x.dimension() - 1; ++i) {
      batch_size = batch_size * x.size(i);
    }
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename T::pointer y_pointer = y.data();
    typename T::difference_type y_batch = y.stride(y.dimension() - 2);
    for (typename T::size_type i = 0; i < batch_size; ++i) {
      cxxblas::swap(x_size, &x_pointer[i * x_batch], &y_pointer[i * y_batch],
                    x_step, y_step);
    }
  } else {
    T x_select = x.select(x.dimension() - 1, 0);
    for (typename T::reference_iterator x_begin = x_select.reference_begin(),
             x_end = x_select.reference_end(); x_begin != x_end; ++x_begin) {
      cxxblas::swap(x_size, &(*x_begin), y[x_begin.position()].data(), x_step,
                    y_step);
    }
  }
  return y;
}

template < typename L >
const typename L::size_tensor& iamax(
    L *l, const typename L::tensor_type &x, const typename L::size_tensor &r) {
  typedef typename L::tensor_type T;
  typedef typename L::size_tensor S;
  if (x.dimension() == 1) {
    if (r.dimension() != 1 || r.size(0) != 1) {
      throw out_of_range("Result dimension does not match.");
    }
  } else if (r.dimension() != x.dimension() - 1) {
    throw out_of_range("Result dimension does not match.");
  } else {
    for (typename T::dim_type i = 0; i < r.dimension(); ++i) {
      if (r.size(i) != x.size(i)) {
        throw out_of_range("Result size does not match.");
      }
    }
  }

  int x_size = static_cast< int >(x.size(x.dimension() - 1));
  int x_step = static_cast< int >(x.stride(x.dimension() - 1));
  if (x.dimension() == 1) {
    typename T::pointer x_pointer = x.data();
    typename S::pointer r_pointer = r.data();
    r_pointer[0] = static_cast< typename S::value_type>(
        cxxblas::iamax(x_size, x_pointer, x_step));
  } else if (x.partialContiguity(0, x.dimension() - 2) &&
             r.partialContiguity(0, r.dimension() - 1)) {
    typename S::size_type batch_size = 1;
    for (typename S::dim_type i = 0; i < r.dimension(); ++i) {
      batch_size = batch_size * r.size(i);
    }
    typename T::pointer x_pointer = x.data();
    typename T::difference_type x_batch = x.stride(x.dimension() - 2);
    typename S::pointer r_pointer = r.data();
    typename S::difference_type r_batch = r.stride(r.dimension() - 1);
    for (typename S::size_type i = 0; i < batch_size; ++i) {
      r_pointer[i * r_batch] = static_cast< typename S::value_type >(
          cxxblas::iamax(x_size, &x_pointer[i * x_batch], x_step));
    }
  } else {
    for (typename S::reference_iterator r_begin = r.reference_begin(),
             r_end = r.reference_end(); r_begin != r_end; ++r_begin) {
      *r_begin = static_cast< typename S::value_type >(
          cxxblas::iamax(x_size, x[r_begin.position()].data(), x_step));
    }
  }
  return r;
}

template < typename L >
typename L::tensor_type* asum(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *r) {
  typedef typename L::tensor_type T;
  if (x.dimension() == 1) {
    r->resize(1);
  } else {
    typename T::size_storage r_size(x.dimension() - 1);
    for (typename T::dim_type i = 0; i < r_size.size(); ++i) {
      r_size[i] = x.size(i);
    }
    r->resize(r_size);
  }
  asum(l, x, *r);
  return r;
}

template < typename L >
typename L::tensor_type* axpy(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *y,
    const typename L::value_type &a) {
  axpy(l, x, y->resizeAs(x).zero(), a);
  return y;
}

template < typename L >
typename L::tensor_type* copy(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *r) {
  copy(l, x, r->resizeAs(x).zero());
  return r;
}

template < typename L >
typename L::tensor_type* dot(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    typename L::tensor_type *r) {
  typedef typename L::tensor_type T;
  if (x.dimension() == 1) {
    r->resize(1);
  } else {
    typename T::size_storage r_size(x.dimension() - 1);
    for (typename T::dim_type i = 0; i < r_size.size(); ++i) {
      r_size[i] = x.size(i);
    }
    r->resize(r_size);
  }
  dot(l, x, y, *r);
  return r;
}

template < typename L >
typename L::tensor_type* dotc(
    L *l, const typename L::tensor_type &x, const typename L::tensor_type &y,
    typename L::tensor_type *r) {
  typedef typename L::tensor_type T;
  if (x.dimension() == 1) {
    r->resize(1);
  } else {
    typename T::size_storage r_size(x.dimension() - 1);
    for (typename T::dim_type i = 0; i < r_size.size(); ++i) {
      r_size[i] = x.size(i);
    }
    r->resize(r_size);
  }
  dotc(l, x, y, *r);
  return r;
}

template < typename L >
typename L::tensor_type* nrm2(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *r) {
  typedef typename L::tensor_type T;
  if (x.dimension() == 1) {
    r->resize(1);
  } else {
    typename T::size_storage r_size(x.dimension() - 1);
    for (typename T::dim_type i = 0; i < r_size.size(); ++i) {
      r_size[i] = x.size(i);
    }
    r->resize(r_size);
  }
  nrm2(l, x, *r);
  return r;
}

template < typename L >
typename L::tensor_type* rot(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *y,
    const typename L::value_type &c, const typename L::value_type &s) {
  rot(l, x, y->resizeAs(x).zero(), c, s);
  return y;
}

template < typename L >
typename L::tensor_type* rotm(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *y,
    const typename L::tensor_type &p) {
  rotm(l, x, y->resizeAs(x).zero(), p);
  return y;
}

template < typename L >
typename L::tensor_type* scal(
    L *l, typename L::tensor_type *x, const typename L::value_type &a) {
  scal(l, x->zero(), a);
  return x;
}

template < typename L >
typename L::tensor_type* swap(
    L *l, const typename L::tensor_type &x, typename L::tensor_type *y) {
  swap(l, x, y->resizeAs(x).zero());
  return y;
}

template < typename L >
typename L::size_tensor* iamax(
    L *l, const typename L::tensor_type &x, typename L::size_tensor *r) {
  if (x.dimension() == 1) {
    r->resize(1);
  } else {
    typename L::size_tensor::size_storage r_size(x.dimension() - 1);
    for (typename L::size_tensor::dim_type i = 0; i < r_size.size(); ++i) {
      r_size[i] = x.size(i);
    }
    r->resize(r_size);
  }
  iamax(l, x, *r);
  return r;
}

}  // namespace math
}  // namespace linalg
}  // namespace thunder

#endif  // THUNDER_LINALG_MATH_INL_BLASLEVEL1_HPP_
