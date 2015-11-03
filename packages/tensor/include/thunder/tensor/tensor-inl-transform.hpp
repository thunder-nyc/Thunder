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
#include "thunder/tensor/complex.hpp"
#include "thunder/tensor/index_iterator.hpp"
#include "thunder/tensor/math.hpp"

namespace thunder {
namespace tensor {

template < typename S >
template < typename T >
Tensor< S > Tensor< S >::viewAs(const T &y, size_type os) const {
  size_storage sz(y.dimension());
  stride_storage st(y.dimension());
  for (dim_type i = 0; i < sz.size(); ++i) {
    sz[i] = static_cast< size_type >(y.size(i));
    st[i] = static_cast< difference_type >(y.stride(i));
  }
  return this->view(sz, st, os);
}

template < typename S >
Tensor< S > Tensor< S >::viewAs(const Tensor &y, size_type os) const {
  return this->view(y.size(), y.stride(), os);
}

template < typename S >
template < typename T >
Tensor< S > Tensor< S >::viewAs(const T &y, stride_storage st,
                                size_type os) const {
  size_storage sz(y.dimension());
  for (dim_type i = 0; i < sz.size(); ++i) {
    sz[i] = static_cast< size_type >(y.size(i));
  }
  return this->view(sz, st, os);
}

template < typename S >
Tensor< S > Tensor< S >::viewAs(const Tensor &y, stride_storage st,
                                size_type os) const {
  return this->view(y.size(), st, os);
}

template < typename S >
template < typename T >
Tensor< S > Tensor< S >::extract(const T &y) const {
  return math::extract(*this, y);
}

template < typename S >
template < typename T >
Tensor< S > Tensor< S >::shuffle(const T &y) const {
  return math::shuffle(*this, y);
}

template < typename S >
template < typename T >
Tensor< S > Tensor< S >::permute(const T &y, dim_type d) const {
  return math::permute(*this, y, d);
}

template < typename S >
template < typename TR >
TR Tensor< S >::getReal(typename TR::allocator_type alloc) const {
  return math::getReal< Tensor, TR >(*this, alloc);
}

template < typename S >
template < typename TR >
TR Tensor< S >::getImag(typename TR::allocator_type alloc) const {
  return math::getImag< Tensor, TR >(*this, alloc);
}

template < typename S >
template < typename TR >
TR Tensor< S >::getArg(typename TR::allocator_type alloc) const {
  return math::getArg< Tensor, TR >(*this, alloc);
}

template < typename S >
template < typename TR >
TR Tensor< S >::getCnrm(typename TR::allocator_type alloc) const {
  return math::getCnrm< Tensor, TR >(*this, alloc);
}


// Static templated subtensor extractors are delegated
template < typename S >
template < typename T >
Tensor< S > Tensor< S >::viewAs(const Tensor &x, const T &y, size_type os) {
  return x.viewAs(y, os);
}

template < typename S >
Tensor< S > Tensor< S >::viewAs(
    const Tensor &x, const Tensor &y, size_type os) {
  return x.viewAs(y, os);
}

template < typename S >
template < typename T >
Tensor< S > Tensor< S >::viewAs(const Tensor &x, const T &y, stride_storage st,
                                size_type os) {
  return x.viewAs(y, st, os);
}

template < typename S >
Tensor< S > Tensor< S >::viewAs(const Tensor &x, const Tensor &y,
                                stride_storage st, size_type os) {
  return x.viewAs(y, st, os);
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

template < typename S >
template < typename TR >
TR Tensor< S >::getReal(const Tensor &x, typename TR::allocator_type alloc) {
  return x.getReal< TR >(alloc);
}

template < typename S >
template < typename TR >
TR Tensor< S >::getImag(const Tensor &x, typename TR::allocator_type alloc) {
  return x.getImag< TR >(alloc);
}

template < typename S >
template < typename TR >
TR Tensor< S >::getArg(const Tensor &x, typename TR::allocator_type alloc) {
  return x.getArg< TR >(alloc);
}

template < typename S >
template < typename TR >
TR Tensor< S >::getCnrm(const Tensor &x, typename TR::allocator_type alloc) {
  return x.getCnrm< TR >(alloc);
}

template < typename S >
Tensor< S > Tensor< S >::narrow(dim_type dim, size_type pos,
                                size_type size) const {
  if (dim >= size_.size()) {
    throw out_of_range("Dimension exceeds limit.");
  }
  if (pos + size > size_[dim]) {
    throw out_of_range("Position and size exceed limit.");
  }
  size_storage sz = size_;
  sz[dim] = size;
  return Tensor(sz, stride_, storage_, offset_ + pos * stride_[dim]);
}

template < typename S >
Tensor< S > Tensor< S >::select(dim_type dim, size_type pos) const {
  if (dim >= size_.size()) {
    throw out_of_range("Dimension exceeds limit.");
  }
  if (pos >= size_[dim]) {
    throw out_of_range("Position exceed limit.");
  }
  if (size_.size() == 1) {
    size_storage sz(1, 1);
    stride_storage st(1, 1);
    return Tensor(sz, st, storage_, offset_ + pos * stride_[dim]);
  } else {
    size_storage sz(size_.size() - 1);
    stride_storage st(stride_.size() - 1);
    for (dim_type i = 0; i < dim; ++i) {
      sz[i] = size_[i];
      st[i] = stride_[i];
    }
    for (dim_type i = dim + 1; i < size_.size(); ++i) {
      sz[i - 1] = size_[i];
      st[i - 1] = stride_[i];
    }
    return Tensor(sz, st, storage_, offset_ + pos * stride_[dim]);
  }
}

template < typename S >
Tensor< S > Tensor< S >::view(size_type sz0) const {
  return Tensor(size_storage({sz0}), storage_, 0);
}

template < typename S >
Tensor< S > Tensor< S >::view(size_type sz0, size_type sz1) const {
  return Tensor(size_storage({sz0, sz1}), storage_, 0);
}

template < typename S >
Tensor< S > Tensor< S >::view(
    size_type sz0, size_type sz1, size_type sz2) const {
  return Tensor(size_storage({sz0, sz1, sz2}), storage_, 0);
}

template < typename S >
Tensor< S > Tensor< S >::view(
    size_type sz0, size_type sz1, size_type sz2, size_type sz3) const {
  return Tensor(size_storage({sz0, sz1, sz2, sz3}), storage_, 0);
}

template < typename S >
Tensor< S > Tensor< S >::view(size_storage sz, size_type os) const {
  return Tensor(sz, storage_, os);
}

template < typename S >
Tensor< S > Tensor< S >::view(size_storage sz, stride_storage st,
                              size_type os) const {
  return Tensor(sz, st, storage_, os);
}

template < typename S >
Tensor< S > Tensor< S >::transpose(dim_type dim0, dim_type dim1) const {
  size_storage sz(size_);
  stride_storage st(stride_);
  ::std::swap(sz[dim0], sz[dim1]);
  ::std::swap(st[dim0], st[dim1]);
  return Tensor(sz, st, storage_, offset_);
}

template < typename S >
Tensor< S > Tensor< S >::unfold(dim_type dim, size_type size,
                                size_type step) const {
  if (dim >= size_.size()) {
    throw out_of_range("Dimension exceeds limit.");
  }
  if (size > size_[dim]) {
    throw out_of_range("Size exceeds limit.");
  }
  size_storage sz(size_.size() + 1);
  stride_storage st(stride_.size() + 1);
  for (dim_type i = 0; i < dim; ++i) {
    sz[i] = size_[i];
    st[i] = stride_[i];
  }
  sz[dim] = (size_[dim] - size) / step + 1;
  st[dim] = stride_[dim] * step;
  sz[dim + 1] = size;
  st[dim + 1] = stride_[dim];
  for (dim_type i = dim + 2; i < sz.size(); ++i) {
    sz[i] = size_[i - 1];
    st[i] = stride_[i - 1];
  }
  return Tensor(sz, st, storage_, offset_);
}

template < typename S >
Tensor< S > Tensor< S >::clone() const {
  return Tensor(size_, stride_, allocator()).copy(*this);
}

template < typename S >
Tensor< S > Tensor< S >::cat(const Tensor &y, dim_type dim) const {
  if (size_.size() != y.dimension()) {
    throw out_of_range("Dimension mismatches.");
  }
  if (dim >= size_.size()) {
    throw out_of_range("Dimension exceeds limit.");
  }
  for (dim_type i = 0; i < dim; ++i) {
    if (size_[i] != y.size(i)) {
      throw out_of_range("Size mismatches.");
    }
  }
  for (dim_type i = dim + 1; i < size_.size(); ++i) {
    if (size_[i] != y.size(i)) {
      throw out_of_range("Size mismatches.");
    }
  }
  size_storage sz(size_);
  sz[dim] = size_[dim] + y.size(dim);
  Tensor t(sz, allocator());
  t.narrow(dim, 0, size_[dim]).copy(*this);
  t.narrow(dim, size_[dim], y.size(dim)).copy(y);
  return t;
}

template < typename S >
Tensor< S > Tensor< S >::reshape(size_type sz0) const {
  if (!partialContiguity(0, size_.size() - 1)) {
    throw contiguity_error(
        "Reshaping is impossible because of non-contiguity.");
  }
  if (sz0 != length()) {
    throw out_of_range("Length mismatches.");
  }
  return Tensor(
      size_storage({sz0}), stride_storage({stride_[stride_.size() - 1]}),
      storage_, offset_);
}

template < typename S >
Tensor< S > Tensor< S >::reshape(size_type sz0, size_type sz1) const {
  if (!partialContiguity(0, size_.size() - 1)) {
    throw contiguity_error(
        "Reshaping is impossible because of non-contiguity.");
  }
  if (sz0 * sz1 != length()) {
    throw out_of_range("Length mismatches.");
  }
  stride_storage st(2);
  st[1] = stride_[stride_.size() - 1];
  st[0] = sz1 * st[1];
  return Tensor(size_storage({sz0, sz1}), st, storage_, offset_);
}

template < typename S >
Tensor< S > Tensor< S >::reshape(size_type sz0, size_type sz1,
                                 size_type sz2) const {
  if (!partialContiguity(0, size_.size() - 1)) {
    throw contiguity_error(
        "Reshaping is impossible because of non-contiguity.");
  }
  if (sz0 * sz1 * sz2 != length()) {
    throw out_of_range("Length mismatches.");
  }
  stride_storage st(3);
  st[2] = stride_[stride_.size() - 1];
  st[1] = sz2 * st[2];
  st[0] = sz1 * st[1];
  return Tensor(size_storage({sz0, sz1, sz2}), st, storage_, offset_);
}

template < typename S >
Tensor< S > Tensor< S >::reshape(size_type sz0, size_type sz1, size_type sz2,
                                 size_type sz3) const {
  if (!partialContiguity(0, size_.size() - 1)) {
    throw contiguity_error(
        "Reshaping is impossible because of non-contiguity.");
  }
  if (sz0 * sz1 * sz2 * sz3 != length()) {
    throw out_of_range("Length mismatches.");
  }
  stride_storage st(4);
  st[3] = stride_[stride_.size() - 1];
  st[2] = sz3 * st[3];
  st[1] = sz2 * st[2];
  st[0] = sz1 * st[1];
  return Tensor(size_storage({sz0, sz1, sz2, sz3}), st, storage_, offset_);
}

template < typename S >
Tensor< S > Tensor< S >::reshape(size_storage sz) const {
  if (!partialContiguity(0, size_.size() - 1)) {
    throw contiguity_error(
        "Reshaping is impossible because of non-contiguity.");
  }
  size_type t_length = 1;
  for (dim_type i = 0; i < sz.size(); ++i) {
    t_length *= sz[i];
  }
  if (t_length != length()) {
    throw out_of_range("Length mismatches.");
  }
  stride_storage st(sz.size());
  st[st.size() - 1] = stride_[stride_.size() - 1];
  for (dim_type i = st.size() - 1; i > 0; --i) {
    st[i - 1] = sz[i] * st[i];
  }
  return Tensor(sz, st, storage_, offset_);
}

template < typename S >
typename Tensor< S >::real_tensor Tensor< S >::viewReal() const {
  return math::viewReal(*this);
}

template < typename S >
typename Tensor< S >::real_tensor Tensor< S >::viewImag() const {
  return math::viewImag(*this);
}

template < typename S >
Tensor< S > Tensor< S >::narrow(const Tensor &x, dim_type dim, size_type pos,
                                size_type size) {
  return x.narrow(dim, pos, size);
}

template < typename S >
Tensor< S > Tensor< S >::select(const Tensor &x, dim_type dim, size_type pos) {
  return x.select(dim, pos);
}

template < typename S >
Tensor< S > Tensor< S >::view(const Tensor &x, size_type sz0) {
  return x.view(sz0);
}

template < typename S >
Tensor< S > Tensor< S >::view(const Tensor &x, size_type sz0, size_type sz1) {
  return x.view(sz0, sz1);
}

template < typename S >
Tensor< S > Tensor< S >::view(const Tensor &x, size_type sz0, size_type sz1,
                              size_type sz2) {
  return x.view(sz0, sz1, sz2);
}

template < typename S >
Tensor< S > Tensor< S >::view(const Tensor &x, size_type sz0, size_type sz1,
                              size_type sz2, size_type sz3) {
  return x.view(sz0, sz1, sz2, sz3);
}

template < typename S >
Tensor< S > Tensor< S >::view(const Tensor &x, size_storage sz, size_type os) {
  return x.view(sz, os);
}

template < typename S >
Tensor< S > Tensor< S >::view(const Tensor &x, size_storage sz,
                              stride_storage st, size_type os) {
  return x.view(sz, st, os);
}

template < typename S >
Tensor< S > Tensor< S >::transpose(const Tensor &x, dim_type dim0,
                                   dim_type dim1) {
  return x.transpose(dim0, dim1);
}

template < typename S >
Tensor< S > Tensor< S >::unfold(const Tensor &x, dim_type dim, size_type size,
                                size_type step) {
  return x.unfold(dim, size, step);
}

template < typename S >
Tensor< S > Tensor< S >::clone(const Tensor& x) {
  return x.clone();
}

template < typename S >
Tensor< S > Tensor< S >::cat(const Tensor &x, const Tensor &y, dim_type dim) {
  return x.cat(y, dim);
}

template < typename S >
Tensor< S > Tensor< S >::reshape(const Tensor &x, size_type sz0) {
  return x.reshape(sz0);
}

template < typename S >
Tensor< S > Tensor< S >::reshape(const Tensor &x, size_type sz0,
                                 size_type sz1) {
  return x.reshape(sz0, sz1);
}

template < typename S >
Tensor< S > Tensor< S >::reshape(const Tensor &x, size_type sz0, size_type sz1,
                                 size_type sz2) {
  return x.reshape(sz0, sz1, sz2);
}

template < typename S >
Tensor< S > Tensor< S >::reshape(const Tensor &x, size_type sz0, size_type sz1,
                      size_type sz2, size_type sz3) {
  return x.reshape(sz0, sz1, sz2, sz3);
}

template < typename S >
Tensor< S > Tensor< S >::reshape(const Tensor &x, size_storage sz) {
  return x.reshape(sz);
}

template < typename S >
typename Tensor< S >::real_tensor Tensor< S >::viewReal(const Tensor &x) {
  return x.viewReal();
}

template < typename S >
typename Tensor< S >::real_tensor Tensor< S >::viewImag(const Tensor &x) {
  return x.viewImag();
}

}  // namespace tensor
}  // namespace thunder

#endif  // THUNDER_TENSOR_TENSOR_INL_TRANSFORM_HPP_
