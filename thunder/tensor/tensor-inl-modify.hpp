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

#ifndef THUNDER_TENSOR_TENSOR_INL_MODIFY_HPP_
#define THUNDER_TENSOR_TENSOR_INL_MODIFY_HPP_

#include "thunder/tensor/tensor.hpp"
#include "thunder/tensor/tensor-inl.hpp"

#include <utility>

#include "thunder/exception.hpp"

namespace thunder {
namespace tensor {

template < typename S >
template < typename T >
Tensor< S >& Tensor< S >::copy(const T &y) {
  if (length() != y.length()) {
    throw length_error("Data have inconsistent length for copy");
  }
  if (isContiguous() && y.isContiguous()) {
    size_type x_length = length();
    pointer x_data = data();
    typename T::pointer y_data = y.data();
    for (size_type i = 0; i < x_length; ++i) {
      x_data[i] = static_cast< value_type >(y_data[i]);
    }
  } else {
    typename T::reference_iterator y_begin = y.reference_begin();
    for (reference_iterator x_begin = reference_begin(),
             x_end = reference_end(); x_begin != x_end; ++x_begin, ++y_begin) {
      *x_begin = static_cast< value_type>(*y_begin);
    }
  }
  return *this;
}

template< typename S >
Tensor< S >& Tensor< S >::copy(const Tensor &y) {
  if (length() != y.length()) {
    throw length_error("Data have inconsistent length for copy");
  }
  if (isContiguous() && y.isContiguous()) {
    size_type x_length = length();
    pointer x_data = data();
    pointer y_data = y.data();
    for (size_type i = 0; i < x_length; ++i) {
      x_data[i] = y_data[i];
    }
  } else {
    reference_iterator y_begin = y.reference_begin();
    for (reference_iterator x_begin = reference_begin(),
             x_end = reference_end(); x_begin != x_end; ++x_begin, ++y_begin) {
      *x_begin = *y_begin;
    }
  }
  return *this;
}

template < typename S >
template < typename T >
Tensor< S >& Tensor< S >::resizeAs(const T &y) {
  size_storage sz(static_cast< dim_type >(y.dimension()));
  for (dim_type i = 0; i < sz.size(); ++i) {
    sz[i] = y.size(static_cast< typename T::dim_type >(i));
  }
  return this->resize(sz);
}

template < typename S >
Tensor< S >& Tensor< S >::resizeAs(const Tensor &y) {
  return this->resize(y.size());
}

// Static non-virtual templated modifiers are delegated
template < typename S >
template < typename T >
Tensor< S >& Tensor< S >::copy(Tensor *x, const T &y) {
  return x->copy(y);
}

template < typename S >
Tensor< S >& Tensor< S >::copy(Tensor *x, const Tensor &y) {
  return x->copy(y);
}

template < typename S >
template < typename T >
Tensor< S >& Tensor< S >::resizeAs(Tensor *x, const T &y) {
  return x->resizeAs(y);
}

template < typename S >
Tensor< S >& Tensor< S >::resizeAs(Tensor *x, const Tensor &y) {
  return x->resizeAs(y);
}

// Normal modifiers
template < typename S >
Tensor< S >& Tensor< S >::set(const Tensor &y) {
  return this->set(y.storage_, y.offset_);
}

template < typename S >
Tensor< S >& Tensor< S >::set(storage_pointer storage, size_type os) {
  difference_type max_offset = os, min_offset = os;
  for (dim_type i = 0; i < size_.size(); ++i) {
    if (stride_[i] < 0) {
      min_offset = min_offset + (size_[i]-1)*stride_[i];
    } else if (stride_[i] > 0) {
      max_offset = max_offset + (size_[i]-1)*stride_[i];
    }
  }
  if (min_offset < 0 || max_offset >= storage_->size()) {
    throw out_of_range("Offset, size and stride exceed storage size.");
  }
  ::std::swap(storage_, storage);
  ::std::swap(offset_, os);
  return *this;
}

template < typename S >
Tensor< S >& Tensor< S >::set(
    size_storage sz, storage_pointer s, size_type os) {
  stride_storage st(sz.size());
  st[st.size() - 1] = 1;
  for (dim_type i = st.size() - 1; i > 0; --i) {
    st[i - 1] = sz[i] * st[i];
  }
  return this->set(sz, st, s, os);
}

template < typename S >
Tensor< S >& Tensor< S >::set(
    size_storage sz, stride_storage st, storage_pointer s, size_type os) {
  if (s == nullptr) {
    throw invalid_argument("Storage is nullptr.");
  }
  if (sz.size() == 0) {
    throw invalid_argument("Size is empty.");
  }
  for (const size_type &size_x : sz) {
    if (size_x == 0) {
      throw invalid_argument("Size evaluates to zero.");
    }
  }
  difference_type max_offset = os, min_offset = os;
  for (dim_type i = 0; i < sz.size(); ++i) {
    if (st[i] < 0) {
      min_offset = min_offset + (sz[i]-1)*st[i];
    } else if (st[i] > 0) {
      max_offset = max_offset + (sz[i]-1)*st[i];
    }
  }
  if (min_offset < 0 || max_offset >= s->size()) {
    throw out_of_range("Offset, size and stride exceed storage size.");
  }
  ::std::swap(size_, sz);
  ::std::swap(stride_, st);
  ::std::swap(storage_, s);
  ::std::swap(offset_, os);
  return *this;
}

template < typename S >
Tensor< S >& Tensor< S >::resize(size_storage sz) {
  stride_storage st(sz.size());
  st[st.size() - 1] = 1;
  for (dim_type i = st.size() - 1; i > 0; --i) {
    st[i - 1] = sz[i] * st[i];
  }
  return resize(sz, st);
}

template < typename S >
Tensor< S >& Tensor< S >::resize(size_storage sz, stride_storage st) {
  if (sz.size() == 0) {
    throw invalid_argument("Size is empty.");
  }
  if (st.size() == 0) {
    throw invalid_argument("Stride is empty.");
  }
  if (sz.size() != st.size()) {
    throw invalid_argument("Size and stride have different length.");
  }
  for (const size_type &size_x : sz) {
    if (size_x == 0) {
      throw invalid_argument("Size evaluates to zero.");
      }
  }
  bool really_resize = false;
  if (size_.size() == sz.size()) {
    for (dim_type i = 0; i < size_.size(); ++i) {
      if (size_[i] != sz[i] || stride_[i] != st[i]) {
        really_resize = true;
        break;
      }
    }
  } else {
    really_resize = true;
  }
  if (really_resize) {
    difference_type max_offset = 0, min_offset = 0;
    for (dim_type i = 0; i < sz.size(); ++i) {
      if (st[i] < 0) {
        min_offset = min_offset + (sz[i]-1)*st[i];
      } else if (st[i] > 0) {
        max_offset = max_offset + (sz[i]-1)*st[i];
      }
    }
    storage_ = ::std::make_shared< S >(max_offset - min_offset + 1);
    offset_ = -min_offset;
    ::std::swap(size_, sz);
    ::std::swap(stride_, st);
  }
  return *this;
}

template < typename S >
Tensor< S >& Tensor< S >::contiguous() {
  if (isContiguous()) {
    return *this;
  }
  return resize(size_);
}

template < typename S >
Tensor< S >& Tensor< S >::squeeze() {
  for (dim_type i = 0; i < size_.size() && size_.size() > 1; ++i) {
    if (size_[i] == 1) {
      size_storage sz(size_.size() - 1);
      stride_storage st(stride_.size() - 1);
      for (dim_type j = 0; j < i; ++j) {
        sz[j] = size_[j];
        st[j] = stride_[j];
      }
      for (dim_type j = i; j < sz.size(); ++j) {
        sz[j] = size_[j + 1];
        st[j] = stride_[j + 1];
      }
      ::std::swap(size_, sz);
      ::std::swap(stride_, st);
    }
  }
  return *this;
}

template < typename S >
Tensor< S >& Tensor< S >::set(Tensor *x, const Tensor &y) {
  return x->set(y);
}

template < typename S >
Tensor< S >& Tensor< S >::set(Tensor *x, storage_pointer s, size_type os) {
  return x->set(s, os);
}

template < typename S >
Tensor< S >& Tensor< S >::set(
    Tensor *x, size_storage sz, storage_pointer s, size_type os) {
  return x->set(sz, s, os);
}

template < typename S >
Tensor< S >& Tensor< S >::set(
    Tensor *x, size_storage sz, stride_storage st, storage_pointer s,
    size_type os) {
  return x->set(sz, st, s, os);
}

template < typename S >
Tensor< S >& Tensor< S >::resize(Tensor *x, size_storage sz) {
  return x->resize(sz);
}

template < typename S >
Tensor< S >& Tensor< S >::resize(
    Tensor *x, size_storage sz, stride_storage st) {
  return x->resize(sz, st);
}

template < typename S >
Tensor< S >& Tensor< S >::contiguous(Tensor *x) {
  return x->contiguous();
}

template < typename S >
Tensor< S >& Tensor< S >::squeeze(Tensor *x) {
  return x->squeeze();
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_MODIFY_HPP_
