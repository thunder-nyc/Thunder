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
Tensor< S >& Tensor< S >::resizeAs(const T &y) {
  size_storage sz(y.dimension());
  for (dim_type i = 0; i < sz.size(); ++i) {
    sz[i] = static_cast< size_type >(y.size(i));
  }
  return this->resize(sz);
}

template < typename S >
template < typename T >
Tensor< S >& Tensor< S >::resizeAs(Tensor *x, const T &y) {
  return x->resizeAs(y);
}

template < typename S >
template < typename T >
Tensor< S >& Tensor< S >::sort(dim_type d, bool r, T *y) {
  return *this;
}

template < typename S >
template < typename T >
Tensor< S >& Tensor< S >::sort(Tensor *x, dim_type d, bool r, T *y) {
  return x->sort(d, r, y);
}

template < typename S >
Tensor< S >& Tensor< S >::set(const Tensor &y) {
  return this->set(y.storage_, y.offset_);
}

template < typename S >
Tensor< S >& Tensor< S >::set(storage_pointer s, size_type os) {
  difference_type max_offset = os, min_offset = os;
  for (dim_type i = 0; i < size_.size(); ++i) {
    if (stride_[i] < 0) {
      min_offset = min_offset + (size_[i]-1)*stride_[i];
    } else if (stride_[i] > 0) {
      max_offset = max_offset + (size_[i]-1)*stride_[i];
    }
  }
  if (min_offset < 0 ||
      max_offset >= static_cast< difference_type >(s->size())) {
    throw out_of_range("Offset, size and stride exceed storage size.");
  }
  ::std::swap(storage_, s);
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
  if (min_offset < 0 ||
      max_offset >= static_cast< difference_type >(s->size())) {
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
Tensor< S >& Tensor< S >::resize(size_type sz) {
  return resize(size_storage({sz}));
}

template < typename S >
Tensor< S >& Tensor< S >::resize(size_type sz0, size_type sz1) {
  return resize(size_storage({sz0, sz1}));
}

template < typename S >
Tensor< S >& Tensor< S >::resize(size_type sz0, size_type sz1, size_type sz2) {
  return resize(size_storage({sz0, sz1, sz2}));
}

template < typename S >
Tensor< S >& Tensor< S >::resize(
    size_type sz0, size_type sz1, size_type sz2, size_type sz3) {
  return resize(size_storage({sz0, sz1, sz2, sz3}));
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
  if (!isContiguous()) {
    Tensor< S > t(size_);
    t.copy(*this);
    ::std::swap(size_, t.size_);
    ::std::swap(stride_, t.stride_);
    ::std::swap(storage_, t.storage_);
    ::std::swap(offset_, t.offset_);
  }
  return *this;
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
Tensor< S >& Tensor< S >::unique() {
  if (!storage_.unique()) {
    Tensor< S > t(size_, stride_);
    t.copy(*this);
    ::std::swap(size_, t.size_);
    ::std::swap(stride_, t.stride_);
    ::std::swap(storage_, t.storage_);
    ::std::swap(offset_, t.offset_);
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
Tensor< S >& Tensor< S >::resize(Tensor *x, size_type sz) {
  return x->resize(sz);
}

template < typename S >
Tensor< S >& Tensor< S >::resize(Tensor *x, size_type sz0, size_type sz1) {
  return x->resize(sz0, sz1);
}

template < typename S >
Tensor< S >& Tensor< S >::resize(
    Tensor *x, size_type sz0, size_type sz1, size_type sz2) {
  return x->resize(sz0, sz1, sz2);
}

template < typename S >
Tensor< S >& Tensor< S >::resize(
    Tensor *x, size_type sz0, size_type sz1, size_type sz2, size_type sz3) {
  return x->resize(sz0, sz1, sz2, sz3);
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

template < typename S >
Tensor< S >& Tensor< S >::unique(Tensor *x) {
  return x->unique();
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_MODIFY_HPP_
