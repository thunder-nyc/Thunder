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

#ifndef THUNDER_LINALG_LINALG_INL_CONSTRUCTOR_HPP_
#define THUNDER_LINALG_LINALG_INL_CONSTRUCTOR_HPP_

#include "thunder/linalg/linalg.hpp"
#include "thunder/linalg/linalg-inl.hpp"

#include "thunder/linalg/math.hpp"

namespace thunder {
namespace linalg {

template < typename T, typename H >
const T& Linalg< T, H >::diag(const T &x, const T &r) {
  return math::diag(this, x, r);
}

template < typename T, typename H >
const T& Linalg< T, H >::eye(const size_storage &s, const T &r) {
  return math::eye(this, s, r);
}

template < typename T, typename H >
const T& Linalg< T, H >::linspace(
    const value_type &a, const value_type &b, const T &r) {
  return math::linspace(this, a, b, r);
}

template < typename T, typename H >
const T& Linalg< T, H >::logspace(
    const value_type &a, const value_type &b, const T &r) {
  return math::logspace(this, a, b, r);
}

template < typename T, typename H >
const T& Linalg< T, H >::tril(const T &x, const T &r) {
  return math::tril(this, x, r);
}

template < typename T, typename H >
const T& Linalg< T, H >::triu(const T &x, const T &r) {
  return math::triu(this, x, r);
}

template < typename T, typename H >
T& Linalg< T, H >::diag(const T &x, T &r) {
  return const_cast< T& >(this->diag(x, const_cast< const T& >(r)));
}

template < typename T, typename H >
T& Linalg< T, H >::eye(const size_storage &s, T &r) {
  return const_cast< T& >(this->eye(s, const_cast< const T& >(r)));
}

template < typename T, typename H >
T& Linalg< T, H >::linspace(const value_type &a, const value_type &b, T &r) {
  return const_cast< T& >(this->linspace(a, b, const_cast< const T& >(r)));
}

template < typename T, typename H >
T& Linalg< T, H >::logspace(const value_type &a, const value_type &b, T &r) {
  return const_cast< T& >(this->logspace(a, b, const_cast< const T& >(r)));
}

template < typename T, typename H >
T& Linalg< T, H >::tril(const T &x, T &r) {
  return const_cast< T& >(this->tril(x, const_cast< const T& >(r)));
}

template < typename T, typename H >
T& Linalg< T, H >::triu(const T &x, T &r) {
  return const_cast< T& >(this->triu(x, const_cast< const T& >(r)));
}

template < typename T, typename H >
T* Linalg< T, H >::diag(const T &x, T *r) {
  return math::diag(this, x, r);
}

template < typename T, typename H >
T* Linalg< T, H >::eye(const typename T::size_storage &s, T *r) {
  return math::eye(this, s, r);
}

template < typename T, typename H >
T* Linalg< T, H >::linspace(
    const value_type &a, const value_type &b, size_type n, T *r) {
  return math::linspace(this, a, b, n, r);
}

template < typename T, typename H >
T* Linalg< T, H >::logspace(
    const value_type &a, const value_type &b, size_type n, T *r) {
  return math::logspace(this, a, b, n, r);
}

template < typename T, typename H >
T* Linalg< T, H >::tril(const T &x, T *r) {
  return math::tril(this, x, r);
}

template < typename T, typename H >
T* Linalg< T, H >::triu(const T &x, T *r) {
  return math::triu(this, x, r);
}

template < typename T, typename H >
T Linalg< T, H >::diag(const T &x) {
  T r(x.allocator());
  return *diag(x, &r);
}

template < typename T, typename H >
T Linalg< T, H >::eye(const typename T::size_storage &s, allocator_type alloc) {
  T r(alloc);
  return *eye(s, &r);
}

template < typename T, typename H >
T Linalg< T, H >::linspace(
    const value_type &a, const value_type &b, size_type n,
    allocator_type alloc) {
  T r(alloc);
  return *linspace(a, b, n, &r);
}

template < typename T, typename H >
T Linalg< T, H >::logspace(
    const value_type &a, const value_type &b, size_type n,
    allocator_type alloc) {
  T r(alloc);
  return *logspace(a, b, n, &r);
}

template < typename T, typename H >
T Linalg< T, H >::tril(const T &x) {
  T r(x.allocator());
  return *tril(x, &r);
}

template < typename T, typename H >
T Linalg< T, H >::triu(const T &x) {
  T r(x.allocator());
  return *triu(x, &r);
}

}  //  namespace linalg
}  //  namespace thunder

#endif  // THUNDER_LINALG_LINALG_INL_CONSTRUCTOR_HPP_
