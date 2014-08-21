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

#ifndef THUNDER_TENSOR_TENSOR_INL_TRANSFORM_HPP_
#define THUNDER_TENSOR_TENSOR_INL_TRANSFORM_HPP_

#include "thunder/tensor/tensor.hpp"
#include "thunder/tensor/tensor-inl.hpp"

#include <utility>

#include "thunder/exception.hpp"
#include "thunder/tensor/index_iteratpr.hpp"

namespace thunder {
namespace tensor {

template < typename S >
template < typename T >
Tensor< S > Tensor< S >::viewAs(const T &y, size_type os = 0) const {
  size_storage sz(y.dimension());
  stride_storage st(y.dimension());
  for (dim_type i = 0; i < sz.size(); ++i) {
    sz[i] = static_cast< size_type >(y.size(i));
    st[i] = static_cast< difference_type >(y.stride(i));
  }
  return this->view(sz, st, os);
}

template < typename S >
template < typename T >
Tensor< S > viewAs(const T &y, const stride_storage &st,
                   size_type os = 0) const {
  size_storage sz(y.dimension());
  for (dim_type i = 0; i < sz.size(); ++i) {
    sz[i] = static_cast< size_type >(y.size(i));
  }
  return this->view(sz, st, os);
}

template < typename S >
template < typename T >
Tensor< S > Tensor< S >::extract(const T &y) const {
  if (dimension() < y.dimension()) {
    throw out_of_range("Dimension exceeds limit");
  }
  dim_type y_dimension = static_cast< dim_type > y.dimension();
  for (i = 0; i < y_dimension; ++i) {
    if (size_[i] != static_cast< size_type >(y.size(i))) {
      throw out_of_range("Size does not match");
    }
  }
  size_storage sz(size_.size() - y_dimension + 1);
  for (dim_type i = y_dimension; i < size_.size(); ++i) {
    sz[i - y_dimension + 1] = size_[i];
  }
  // Get the size of returning tensor
  if (y.isContiguous()) {
    y::size_type y_length = y.length();
    y::pointer y_data = y.data();
    for (y::size_type i = 0; i < y_length; ++y) {
      if (static_cast< bool >(y_data[i]) == true) {
        ++sz[0];
      }
    }
  } else {
    for (T::reference_iterator begin = y.reference_begin(),
             end = y.reference_end(); begin != end; ++begin) {
      if (static_cast< bool >(*begin) == true) {
        ++sz[0];
      }
    }
  }
  Tensor< S > t(sz);
  // Really do the copy
  if (isContiguous() && y.isContiguous()) {
    difference_type st = stride_[y_dimension - 1];
    pointer t_data = t.data();
    pointer dt = data();
    size_type current = 0;
    for (y::size_type i = 0; i < y_length; ++y) {
      if (static_cast< bool >(y_data[i]) == true) {
        for (size_type j = 0; j < st; ++j) {
          t_data[current++] = dt[i * st + j];
        }
      }
    }
  } else {
    for (SizeIterator< T::size_storage > begin =
             SizeIterator< T::size_storage >::begin(y.size()),
         end = SizeIterator< T::size_storage >::end(y.size());
         begin != end; ++begin) {
      size_type pos = 0;
      if (static_cast< bool >(y[begin]()) == true) {
        t[pos].copy((*this)[begin]);
      }
    }
  }
  return t;
}

template < typename S >
template < typename T >
Tensor< S > Tensor< S >::shuffle(const T &y) const {
  if (static_cast< size_type >(y.size(y.dimension() - 1)) != size_.size()) {
    throw out_of_range("Shuffle index tensor mismatches");
  }
  if (y.size() == 1) {
    Tensor< S > t;
    size_storage ind(size_.size());
    for (T::dim_type i = 0; i < y.size(); ++i) {
      ind[i] = static_cast< size_type >(y(i));
      if (ind[i] >= size_[i]) {
        throw out_of_range("Shuffle index exceeds limit");
      }
    }
    t() = (*this)(ind);
    return t;
  }
  size_storage sz(y.dimension() - 1);
  for (dim_type i = 0; i < sz.size(); ++i) {
    sz[i] = y.size(i);
  }
  Tensor< S > t(sz);
  size_storage ind(size_.size());
  for (SizeIterator< size_storage > begin =
           SizeIterator< size_storage >::begin(sz),
           end = SizeIterator< size_storage >::end(sz); begin != end; ++begin) {
    for (dim_type i = 0; i < ind.size(); ++i) {
      ind[i] = static_cast< size_type >(y[*begin](i));
      if (ind[i] >= sz[i]) {
        throw out_of_range("Shuffle index exceedds limit");
      }
      t[*begin] = (*this)(ind);
    }
  }
  return t;
}


// Static templated subtensor extractors are delegated
template < typename S >
template < typename T >
Tensor< S > Tensor< S >::viewAs(const Tensor &x, const T &y, size_type os = 0) {
  return x.viewAs(y, os);
}

template < typename S >
template < typename T >
Tensor< S > Tensor< S >::viewAs(const Tensor &x, const T &y,
                                const stride_storage &st, size_type os = 0) {
  return x.viewAs(y, os);
}

template < typename S >
template < typename T >
Tensor< S > Tensor< S >::extract(const Tensor &x, const T &y) {
  return x.extract(y);
}

template < typename S >
template < typename T >
Tensor< S > Tensor< S >::shuffle(const Tensor &x, const T &y) {
  return x.shuffle(y);
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_TRANSFORM_HPP_
