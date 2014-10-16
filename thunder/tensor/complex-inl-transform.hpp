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

#ifndef THUNDER_TENSOR_COMPLEX_INL_TRANSFORM_HPP_
#define THUNDER_TENSOR_COMPLEX_INL_TRANSFORM_HPP_

#include "thunder/tensor/complex.hpp"
#include "thunder/tensor/complex-inl.hpp"

#include <complex>

#include "thunder/tensor/math.hpp"
#include "thunder/tensor/tensor.hpp"

namespace thunder {
namespace tensor {
namespace math {

// Transformations
template < typename D, typename A, typename T1 >
T1 extract(const T1 &x, const Tensor< Storage< ::std::complex< D >, A > > &y) {
  typedef Tensor< Storage < ::std::complex< D >, A > > T2;
  if (x.dimension() < y.dimension()) {
    throw out_of_range("Dimension exceeds limit.");
  }
  typename T1::dim_type y_dimension =
      static_cast< typename T1::dim_type >(y.dimension());
  for (typename T1::dim_type i = 0; i < y_dimension; ++i) {
    if (x.size(i) != static_cast< typename T1::size_type >(y.size(i))) {
      throw out_of_range("Size does not match.");
    }
  }
  typename T1::size_storage sz(x.dimension() - y_dimension + 1);
  for (typename T1::dim_type i = y_dimension; i < x.dimension(); ++i) {
    sz[i - y_dimension + 1] = x.size(i);
  }
  // Get the size of returning tensor
  sz[0] = 0;
  if (y.isContiguous()) {
    typename T2::size_type y_length = y.length();
    typename T2::pointer y_data = y.data();
    for (typename T2::size_type i = 0; i < y_length; ++i) {
      if (static_cast< bool >(::std::real(y_data[i])) == true) {
        ++sz[0];
      }
    }
  } else {
    for (typename T2::reference_iterator begin = y.reference_begin(),
             end = y.reference_end(); begin != end; ++begin) {
      if (static_cast< bool >(::std::real(*begin)) == true) {
        ++sz[0];
      }
    }
  }

  // Create the new tensor and do the copy
  T1 t(sz);
  if (x.isContiguous() && y.isContiguous()) {
    typename T2::size_type y_length = y.length();
    typename T2::pointer y_data = y.data();
    typename T1::difference_type x_step = x.stride(y_dimension - 1);
    typename T1::pointer t_data = t.data();
    typename T1::pointer x_data = x.data();
    typename T1::size_type current = 0;
    for (typename T2::size_type i = 0; i < y_length; ++i) {
      if (static_cast< bool >(::std::real(y_data[i])) == true) {
        for (typename T1::size_type j = 0; j < x_step; ++j) {
          t_data[current++] = x_data[i * x_step + j];
        }
      }
    }
  } else {
    typename T1::size_storage y_size(y.dimension());
    for (typename T1::dim_type i = 0; i < y_size.size(); ++i) {
      y_size[i] = y.size(i);
    }
    typename T1::size_type pos = 0;
    IndexIterator< typename T1::size_storage > t_begin =
        IndexIterator< typename T1::size_storage>::begin(y_size);
    for (IndexIterator< typename T2::size_storage > begin =
             IndexIterator< typename T2::size_storage >::begin(y.size()),
             end = IndexIterator< typename T2::size_storage >::end(y.size());
         begin != end; ++begin, ++t_begin) {
      if (static_cast< bool >(::std::real(y[*begin]())) == true) {
        t[pos++].copy(x[*t_begin]);
      }
    }
  }
  return t;
}

template < typename D, typename A, typename T1 >
T1 shuffle(const T1 &x, const Tensor< Storage< ::std::complex< D >, A > > &y) {
  typedef Tensor< Storage< ::std::complex< D >, A > > T2;
  if (static_cast< typename T1::dim_type >(
          y.size(y.dimension() - 1)) != x.dimension()) {
    throw out_of_range("Shuffle dimension mismatches.");
  }
  if (y.dimension() == 1) {
    T1 t;
    typename T1::size_storage ind(x.dimension());
    for (typename T2::dim_type i = 0; i < y.size(0); ++i) {
      ind[i] = static_cast< typename T1::size_type >(::std::real(y(i)));
      if (ind[i] >= x.size(i)) {
        throw out_of_range("Shuffle index exceeds limit.");
      }
    }
    t() = x(ind);
    return t;
  }
  typename T1::size_storage sz(y.dimension() - 1);
  for (typename T1::dim_type i = 0; i < sz.size(); ++i) {
    sz[i] = y.size(i);
  }
  T1 t(sz);
  typename T1::size_storage ind(x.dimension());
  for (IndexIterator< typename T1::size_storage > begin =
           IndexIterator< typename T1::size_storage >::begin(sz),
           end = IndexIterator< typename T1::size_storage >::end(sz);
       begin != end; ++begin) {
    for (typename T1::dim_type i = 0; i < ind.size(); ++i) {
      ind[i] = static_cast< typename T1::size_type >(::std::real(y[*begin](i)));
      if (ind[i] >= x.size(i)) {
        throw out_of_range("Shuffle index exceeds limit.");
      }
    }
    t(*begin) = x(ind);
  }
  return t;
}

}  // namespace math
}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_COMPLEX_INL_TRANSFORM_HPP_
